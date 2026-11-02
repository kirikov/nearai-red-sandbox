//! copyIn and copyOut, built on `Exec` so any backend can reuse them.
//!
//! copyIn trusts the host and its inputs. copyOut trusts nothing the guest
//! says: sizes and hashes only check the transport, and the archive goes
//! through the hostile-tar extractor before anything lands on the host.

use std::path::Path;
use std::process::Command;

use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use sha2::{Digest, Sha256};

use crate::files::{guest_write, script};
use crate::protocol::new_nonce;
use crate::sandbox::{CopyOutResult, Error, Exec, ExecOptions, Result};
use crate::untar::{ExtractLimits, TarRejected, extract_tar};

#[derive(Debug, Clone, Copy)]
pub struct CopyLimits {
    pub max_bytes: u64,
    pub max_entries: usize,
    /// Raw bytes per copyOut read; base64 of this stays under mvm's
    /// ~950 KB silent stdout truncation.
    pub chunk_bytes: u64,
}

impl Default for CopyLimits {
    fn default() -> Self {
        Self { max_bytes: 256 * 1024 * 1024, max_entries: 10_000, chunk_bytes: 524_288 }
    }
}

fn long() -> ExecOptions {
    ExecOptions::new(300_000, 16_384)
}

// -o: never restore the archive's owner; a guest without CAP_CHOWN (the
// container backend) would otherwise fail on every entry.
const UNPACK: &str = r#"mkdir -p -- "$1" && tar -xof "$2" -C "$1"; s=$?; rm -f -- "$2"; exit $s"#;

const PACK: &str = r#"d=$1; t=$2
if [ ! -d "$d" ]; then echo NONE; exit 0; fi
if [ -z "$(ls -A -- "$d")" ]; then echo NONE; exit 0; fi
rm -f -- "$t"
tar -cf "$t" -C "$d" . || exit 3
printf 'TAR %s %s\n' "$(wc -c < "$t")" "$(sha256sum < "$t" | cut -d ' ' -f 1)""#;

const SLICE: &str = r#"dd if="$1" bs="$2" skip="$3" count=1 2>/dev/null | base64"#;
const CLEAN: &str = r#"rm -f -- "$1""#;

fn stderr_of(r: &crate::sandbox::ExecResult) -> String {
    let s = String::from_utf8_lossy(&r.stderr).trim().to_string();
    if s.is_empty() { format!("exit {}", r.exit_code) } else { s }
}

pub fn copy_in_via(sb: &dyn Exec, host_dir: &Path, guest_dir: &str) -> Result<()> {
    // ustar keeps the archive readable by busybox tar (no pax or xattr headers).
    let out = Command::new("tar")
        .args(["--format=ustar", "-cf", "-", "-C"])
        .arg(host_dir)
        .arg(".")
        .env("COPYFILE_DISABLE", "1")
        .output()
        .map_err(|e| Error::Other(format!("copyIn: host tar failed to run: {e}")))?;
    if !out.status.success() {
        return Err(Error::Other(format!("copyIn: host tar failed: {}", String::from_utf8_lossy(&out.stderr).trim())));
    }
    let tmp = format!("/tmp/rts-in-{}.tar", new_nonce());
    guest_write(sb, &tmp, &out.stdout)?;
    let r = sb.exec(&script(UNPACK, &[guest_dir, &tmp]), &long())?;
    if r.exit_code != 0 {
        return Err(Error::Other(format!("copyIn: guest tar failed: {}", stderr_of(&r))));
    }
    Ok(())
}

pub fn copy_out_via(sb: &dyn Exec, guest_dir: &str, host_dir: &Path, limits: &CopyLimits) -> Result<CopyOutResult> {
    let tmp = format!("/tmp/rts-out-{}.tar", new_nonce());
    let result = copy_out_inner(sb, guest_dir, host_dir, limits, &tmp);
    let _ = sb.exec(&script(CLEAN, &[&tmp]), &ExecOptions::new(60_000, 1024));
    result
}

fn copy_out_inner(sb: &dyn Exec, guest_dir: &str, host_dir: &Path, limits: &CopyLimits, tmp: &str) -> Result<CopyOutResult> {
    let packed = sb.exec(&script(PACK, &[guest_dir, tmp]), &long())?;
    if packed.exit_code != 0 {
        return Err(Error::Other(format!("copyOut: guest tar failed: {}", stderr_of(&packed))));
    }
    let reply = String::from_utf8_lossy(&packed.stdout).trim().to_string();
    if reply == "NONE" {
        return Ok(CopyOutResult::NothingToExport);
    }
    let malformed = || Error::Other("copyOut: malformed reply from guest".into());
    let mut parts = reply.split_whitespace();
    let (tag, size, digest) = (parts.next(), parts.next(), parts.next());
    if tag != Some("TAR") || parts.next().is_some() {
        return Err(malformed());
    }
    let size: u64 = size.filter(|s| s.len() <= 16).and_then(|s| s.parse().ok()).ok_or_else(malformed)?;
    let digest = digest.filter(|d| d.len() == 64 && d.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())).ok_or_else(malformed)?;
    // Headers and padding on top of the content cap.
    let overhead = 512 * (2 * limits.max_entries as u64 + 2) + 20_480;
    if size > limits.max_bytes + overhead {
        return Err(TarRejected { entry: "(archive)".into(), reason: format!("size cap of {} bytes exceeded", limits.max_bytes) }.into());
    }

    let mut archive = Vec::with_capacity(size as usize);
    let chunks = size.div_ceil(limits.chunk_bytes);
    // base64 grows 4/3, plus a newline every 76 characters.
    let cap = (limits.chunk_bytes * 4).div_ceil(3) + limits.chunk_bytes.div_ceil(57) + 64;
    for i in 0..chunks {
        let r = sb.exec(&script(SLICE, &[tmp, &limits.chunk_bytes.to_string(), &i.to_string()]), &ExecOptions::new(300_000, cap))?;
        if r.exit_code != 0 || r.truncated {
            return Err(Error::Other(format!("copyOut: reading slice {i} failed")));
        }
        let text: Vec<u8> = r.stdout.iter().copied().filter(|b| !b.is_ascii_whitespace()).collect();
        let bytes = B64.decode(text).map_err(|_| Error::Other(format!("copyOut: slice {i} is not base64")))?;
        let want = limits.chunk_bytes.min(size - i * limits.chunk_bytes);
        if bytes.len() as u64 != want {
            return Err(Error::Other(format!("copyOut: slice {i} is {} bytes, expected {want}", bytes.len())));
        }
        archive.extend(bytes);
    }
    let got: String = Sha256::digest(&archive).iter().map(|b| format!("{b:02x}")).collect();
    if got != digest {
        return Err(Error::Other("copyOut: archive sha256 does not match what the guest reported".into()));
    }
    let x = extract_tar(&archive, host_dir, ExtractLimits { max_bytes: limits.max_bytes, max_entries: limits.max_entries })?;
    Ok(CopyOutResult::Exported { files: x.files, bytes: x.bytes })
}

#[cfg(test)]
mod tests {
    //! The copy scripts run by the host's sh and tar against temp directories
    //! standing in for the guest. GNU tar here, like busybox tar, writes GNU
    //! long-name headers, so the hostile cases match the guest's.

    use super::*;
    use crate::testutil::{HostShell, tempdir};
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;

    const SMALL: CopyLimits = CopyLimits { max_bytes: 1024 * 1024, max_entries: 100, chunk_bytes: 4096 };

    struct Env {
        root: PathBuf,
        guest: PathBuf,
        host: PathBuf,
        sentinel: PathBuf,
    }
    impl Drop for Env {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }
    fn env() -> Env {
        let root = tempdir("copy");
        let sentinel = root.join("sentinel");
        fs::write(&sentinel, "untouched").unwrap();
        Env { guest: root.join("guest-out"), host: root.join("host/out"), sentinel, root }
    }
    fn g(e: &Env) -> String {
        e.guest.to_string_lossy().into_owned()
    }

    #[test]
    fn copy_in_moves_a_nested_tree() {
        let e = env();
        let src = e.root.join("inputs");
        fs::create_dir_all(src.join("a/b")).unwrap();
        let big: Vec<u8> = (0..300 * 1024u32).map(|i| (i % 253) as u8).collect();
        fs::write(src.join("a/b/big.bin"), &big).unwrap();
        fs::write(src.join("top.txt"), "top").unwrap();
        let dest = e.root.join("guest-in");
        copy_in_via(&HostShell::default(), &src, &dest.to_string_lossy()).unwrap();
        assert_eq!(fs::read(dest.join("a/b/big.bin")).unwrap(), big);
        assert_eq!(fs::read_to_string(dest.join("top.txt")).unwrap(), "top");
    }

    #[test]
    fn copy_out_is_byte_exact_across_slices() {
        let e = env();
        fs::create_dir_all(e.guest.join("sub")).unwrap();
        let blob: Vec<u8> = (0..50_000u32).map(|i| (i * 31 % 256) as u8).collect();
        fs::write(e.guest.join("sub/blob"), &blob).unwrap();
        fs::write(e.guest.join("hello.txt"), "hi").unwrap();
        let r = copy_out_via(&HostShell::default(), &g(&e), &e.host, &SMALL).unwrap();
        assert_eq!(r, CopyOutResult::Exported { files: 2, bytes: 50_002 });
        assert_eq!(fs::read(e.host.join("sub/blob")).unwrap(), blob);
    }

    #[test]
    fn missing_or_empty_out_is_nothing_to_export() {
        let e = env();
        assert_eq!(copy_out_via(&HostShell::default(), &g(&e), &e.host, &SMALL).unwrap(), CopyOutResult::NothingToExport);
        fs::create_dir(&e.guest).unwrap();
        assert_eq!(copy_out_via(&HostShell::default(), &g(&e), &e.host, &SMALL).unwrap(), CopyOutResult::NothingToExport);
        assert!(!e.host.exists());
    }

    fn hostile(e: &Env, reason: &str, limits: &CopyLimits) {
        let err = copy_out_via(&HostShell::default(), &g(e), &e.host, limits).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)), "{err}");
        assert!(err.to_string().contains(reason), "{err}");
        assert!(!e.host.exists());
        assert_eq!(fs::read_to_string(&e.sentinel).unwrap(), "untouched");
    }

    #[test]
    fn a_symlink_to_a_host_path_is_refused() {
        let e = env();
        fs::create_dir(&e.guest).unwrap();
        symlink(&e.sentinel, e.guest.join("evil")).unwrap();
        hostile(&e, "unsupported tar entry type 2", &SMALL);
    }

    #[test]
    fn a_hardlink_is_refused() {
        let e = env();
        fs::create_dir(&e.guest).unwrap();
        fs::write(e.guest.join("a"), "a").unwrap();
        fs::hard_link(e.guest.join("a"), e.guest.join("b")).unwrap();
        hostile(&e, "unsupported tar entry type 1", &SMALL);
    }

    #[test]
    fn a_long_name_is_refused() {
        let e = env();
        fs::create_dir(&e.guest).unwrap();
        fs::write(e.guest.join("n".repeat(120)), "x").unwrap();
        hostile(&e, "unsupported tar entry type L", &SMALL);
    }

    #[test]
    fn an_oversized_file_is_refused() {
        let e = env();
        fs::create_dir(&e.guest).unwrap();
        fs::write(e.guest.join("huge"), vec![0u8; 200_000]).unwrap();
        hostile(&e, "size cap", &CopyLimits { max_bytes: 100_000, ..SMALL });
    }
}
