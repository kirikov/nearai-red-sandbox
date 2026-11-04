//! Container backend: the same `Sandbox` seam as mvm, implemented over the
//! Docker CLI.
//!
//! Built for the Linux runner inside a confidential VM, where nested
//! virtualization is unavailable and the wall between jobs is the container
//! runtime (`sysbox-runc` where the daemon offers it, `runc` otherwise) plus
//! the TEE around the whole VM. Everything above this module is unchanged:
//! the guest wrapper protocol, the tool bridge, copyIn/copyOut and the
//! hostile-tar extractor all run against `Exec` and never see the backend.
//!
//! Differences from mvm, deliberately:
//! - No network sessions. A container here has no network at all; a
//!   `CreateSpec` with egress is refused rather than approximated.
//! - No exec serialization. Docker does not journal execs, and a nonzero
//!   guest exit does not poison the container, so calls may overlap.
//! - The image may be a repo digest (`name@sha256:…`) or a local image id
//!   (`sha256:…`). Both are content-addressed; a tag is still refused.

use std::io;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::time::Duration;

use crate::copy::{CopyLimits, copy_in_via, copy_out_via};
use crate::mvm::{
    BACKSTOP, RunOutput, Runner, StateDir, is_image_ref, is_machine_name, run_captured, tail,
};
use crate::protocol::{WrapOptions, new_nonce, parse_result, wrap_argv};
use crate::sandbox::{
    CopyOutResult, CreateSpec, Error, Exec, ExecOptions, ExecResult, Result, Sandbox, sv,
};

const CONTROL_TIMEOUT: Duration = Duration::from_secs(180);
/// `docker start` and `docker rm -f` of a sysbox-runc container can take
/// minutes on a large image: without idmapped mounts sysbox chowns the whole
/// rootfs on the first start (a 650 MB, 35k-file image needed more than 180 s
/// on dstack, 2026-11-04). They get a longer budget than the other controls.
const START_TIMEOUT: Duration = Duration::from_secs(900);

/// Processes a container may hold. The guest wrapper spawns a handful per
/// tool call; a repository's test suite can fan out, but a fork bomb must
/// hit this before it hits the VM.
pub const PIDS_LIMIT: u32 = 1024;

/// PID 1 of the container. `sh` reaps the children it spawned; `--init`
/// reaps the rest.
const KEEPALIVE: [&str; 3] = [
    "sh",
    "-c",
    "trap 'exit 0' TERM INT; while :; do sleep 2147483647; done",
];

/// Runs the `docker` CLI. Which daemon answers is the CLI's business
/// (`DOCKER_HOST` or the default socket).
pub struct DockerRunner {
    pub docker: PathBuf,
}

impl Runner for DockerRunner {
    fn run(&self, argv: &[String], timeout: Duration) -> io::Result<RunOutput> {
        let mut cmd = Command::new(&self.docker);
        cmd.args(argv);
        run_captured(cmd, timeout)
    }
}

/// `sha256:<64 lowercase hex>`: a local image id, as `docker build -q`
/// prints it and `docker image inspect` reports it.
pub fn is_image_id(image: &str) -> bool {
    match image.strip_prefix("sha256:") {
        Some(d) => {
            d.len() == 64
                && d.bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        }
        None => false,
    }
}

/// A content-addressed image reference: repo digest or image id.
pub fn is_container_image(image: &str) -> bool {
    is_image_ref(image) || is_image_id(image)
}

/// A runtime name as `docker info` lists it: letters, digits, `.`, `_`, `-`.
pub fn is_runtime_name(rt: &str) -> bool {
    !rt.is_empty()
        && rt.len() <= 64
        && rt
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
}

pub struct DockerSandbox {
    pub name: String,
    run: Box<dyn Runner>,
    runtime: Option<String>,
    init: bool,
    /// Shared with the launcher and the other tool processes of the
    /// session: the unavailable marker and the exec lock (held shared).
    state: Option<StateDir>,
    dead: Mutex<Option<String>>,
    copy_limits: CopyLimits,
}

impl DockerSandbox {
    pub fn new(name: &str, run: Box<dyn Runner>) -> Result<Self> {
        if !is_machine_name(name) {
            return Err(Error::Other(format!(
                "container name must match rts-<12 hex chars>: {name:?}"
            )));
        }
        Ok(Self {
            name: name.into(),
            run,
            runtime: None,
            init: true,
            state: None,
            dead: Mutex::new(None),
            copy_limits: CopyLimits::default(),
        })
    }

    /// Share the session's state directory: a marker set by any process
    /// refuses every later exec here, and `quiesce` waits for this
    /// sandbox's active execs.
    pub fn with_state(mut self, dir: StateDir) -> Self {
        self.state = Some(dir);
        self
    }

    /// `--runtime` for `docker create`; `None` is the daemon's default.
    pub fn with_runtime(mut self, runtime: Option<String>) -> Result<Self> {
        if let Some(rt) = &runtime
            && !is_runtime_name(rt)
        {
            return Err(Error::Other(format!("runtime name is not plain: {rt:?}")));
        }
        self.runtime = runtime;
        Ok(self)
    }

    /// Whether `docker create` gets `--init`. On by default; off for a
    /// daemon without `docker-init`.
    pub fn with_init(mut self, init: bool) -> Self {
        self.init = init;
        self
    }

    pub fn with_copy_limits(mut self, l: CopyLimits) -> Self {
        self.copy_limits = l;
        self
    }

    pub fn unavailable(&self) -> bool {
        self.dead.lock().unwrap().is_some()
    }

    fn dead_reason(&self) -> Option<String> {
        if let Some(r) = self.dead.lock().unwrap().clone() {
            return Some(r);
        }
        self.state.as_ref().and_then(|s| s.dead_reason())
    }

    fn mark_unavailable(&self, reason: &str) -> Error {
        let mut d = self.dead.lock().unwrap();
        if d.is_none() {
            *d = Some(
                self.state
                    .as_ref()
                    .and_then(|s| s.dead_reason())
                    .unwrap_or_else(|| reason.into()),
            );
        }
        if let Some(s) = &self.state {
            s.mark(reason);
        }
        Error::Unavailable(d.clone().unwrap())
    }

    fn control(&self, argv: &[String], what: &str) -> Result<RunOutput> {
        self.control_with(argv, what, CONTROL_TIMEOUT)
    }

    fn control_with(&self, argv: &[String], what: &str, timeout: Duration) -> Result<RunOutput> {
        let r = self
            .run
            .run(argv, timeout)
            .map_err(|e| Error::Other(format!("{what} failed: {e}")))?;
        if r.killed {
            return Err(Error::Other(format!("{what} timed out")));
        }
        if r.code != Some(0) {
            let t = tail(&r.stderr);
            return Err(Error::Other(format!(
                "{what} failed: {}",
                if t.is_empty() {
                    format!("exit {:?}", r.code)
                } else {
                    t
                }
            )));
        }
        Ok(r)
    }

    /// The `docker create` argv for `spec`, after validation.
    pub fn create_argv(&self, spec: &CreateSpec) -> Result<Vec<String>> {
        if spec.name != self.name {
            return Err(Error::Other(format!(
                "create name {} does not match sandbox {}",
                spec.name, self.name
            )));
        }
        if !is_container_image(spec.image) {
            return Err(Error::Other(format!(
                "image must be a repo digest or an image id: {:?}",
                spec.image
            )));
        }
        if spec.cpus < 1 || spec.memory_mib < 64 {
            return Err("cpus must be at least 1 and memory at least 64 MiB".into());
        }
        if spec.egress.is_some() {
            return Err("network sessions are not supported by the container backend".into());
        }
        let mem = format!("{}m", spec.memory_mib);
        let mut argv = sv(&[
            "create",
            "--name",
            &self.name,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            &PIDS_LIMIT.to_string(),
            "--memory",
            &mem,
            "--memory-swap",
            &mem,
            "--cpus",
            &spec.cpus.to_string(),
            "--tmpfs",
            &format!("/tmp:rw,exec,nosuid,size={mem}"),
            "--workdir",
            "/",
        ]);
        if self.init {
            argv.push("--init".into());
        }
        if let Some(rt) = &self.runtime {
            argv.extend(sv(&["--runtime", rt]));
        }
        argv.push(spec.image.into());
        argv.extend(sv(&KEEPALIVE));
        Ok(argv)
    }
}

fn is_gone(stderr: &[u8]) -> bool {
    let s = String::from_utf8_lossy(stderr);
    s.contains("No such container") || s.contains("is not running") || s.contains("is restarting")
}

impl Exec for DockerSandbox {
    fn exec(&self, argv: &[String], o: &ExecOptions) -> Result<ExecResult> {
        if let Some(r) = self.dead_reason() {
            return Err(Error::Unavailable(r));
        }
        // Execs run concurrently (Docker journals nothing), but teardown's
        // exclusive quiesce still waits for every active one.
        let _shared = match &self.state {
            Some(s) => Some(
                s.lock_shared()
                    .map_err(|e| Error::Other(format!("exec lock: {e}")))?,
            ),
            None => None,
        };
        if let Some(r) = self.dead_reason() {
            return Err(Error::Unavailable(r));
        }
        let nonce = new_nonce();
        let timeout_sec = o.timeout_ms.div_ceil(1000).max(1);
        let wrapped = wrap_argv(
            argv,
            &WrapOptions {
                nonce: &nonce,
                timeout_sec,
                max_output_bytes: o.max_output_bytes,
                workdir: o.workdir.as_deref(),
            },
        )
        .map_err(Error::Other)?;
        let mut full = sv(&["exec", &self.name]);
        full.extend(wrapped);
        let r = match self
            .run
            .run(&full, Duration::from_secs(timeout_sec) + BACKSTOP)
        {
            Ok(r) => r,
            Err(e) => return Err(self.mark_unavailable(&format!("docker exec failed to run: {e}"))),
        };
        if let Some(r) = self.dead_reason() {
            return Err(Error::Unavailable(r)); // removed mid-exec
        }
        if r.killed {
            return Err(self.mark_unavailable("exec exceeded the host backstop; container is lost"));
        }
        match (parse_result(&r.stdout, &r.stderr, &nonce), r.code) {
            (Some(p), Some(0)) => Ok(p),
            (_, code) if is_gone(&r.stderr) => Err(self.mark_unavailable(&format!(
                "container is gone (docker exit {code:?}): {}",
                tail(&r.stderr)
            ))),
            (None, code) => Err(self.mark_unavailable(&format!(
                "exec returned no valid trailer (docker exit {code:?})"
            ))),
            (Some(_), code) => Err(self.mark_unavailable(&format!("docker exec exited {code:?}"))),
        }
    }
}

impl Sandbox for DockerSandbox {
    fn create(&self, spec: &CreateSpec) -> Result<()> {
        let argv = self.create_argv(spec)?;
        if let Some(r) = self.dead_reason() {
            return Err(Error::Unavailable(r));
        }
        self.control(&argv, "docker create")?;
        self.control_with(&sv(&["start", &self.name]), "docker start", START_TIMEOUT)?;
        Ok(())
    }

    fn copy_in(&self, host_dir: &std::path::Path, guest_dir: &str) -> Result<()> {
        copy_in_via(self, host_dir, guest_dir)
    }

    /// Treats the guest as hostile; see copy.rs and untar.rs.
    fn copy_out(&self, guest_dir: &str, host_dir: &std::path::Path) -> Result<CopyOutResult> {
        copy_out_via(self, guest_dir, host_dir, &self.copy_limits)
    }

    fn remove(&self) -> Result<()> {
        self.mark_unavailable("sandbox removed");
        let rm = self
            .run
            .run(&sv(&["rm", "-f", &self.name]), START_TIMEOUT);
        let ps = self.control(
            &sv(&[
                "ps",
                "-a",
                "--filter",
                &format!("name=^/{}$", self.name),
                "--format",
                "{{.Names}}",
            ]),
            "docker ps after rm",
        )?;
        if String::from_utf8_lossy(&ps.stdout)
            .split_whitespace()
            .any(|n| n == self.name)
        {
            let detail = match rm {
                Ok(r) if r.code != Some(0) => format!(" (rm: {})", tail(&r.stderr)),
                Err(e) => format!(" (rm: {e})"),
                _ => String::new(),
            };
            return Err(Error::Other(format!(
                "container still present after rm: {}{detail}",
                self.name
            )));
        }
        Ok(())
    }
}

/// Every `rts-*` container the daemon knows, running or not. For the
/// orphan sweep at session start.
pub fn list_containers(run: &dyn Runner) -> Result<Vec<String>> {
    let r = run
        .run(
            &sv(&[
                "ps",
                "-a",
                "--filter",
                "name=^/rts-",
                "--format",
                "{{.Names}}",
            ]),
            CONTROL_TIMEOUT,
        )
        .map_err(|e| Error::Other(format!("docker ps failed: {e}")))?;
    if r.code != Some(0) {
        return Err(Error::Other(format!(
            "docker ps failed: {}",
            tail(&r.stderr)
        )));
    }
    Ok(String::from_utf8_lossy(&r.stdout)
        .split_whitespace()
        .filter(|n| is_machine_name(n))
        .map(String::from)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::WRAPPER;
    use crate::sandbox::Egress;
    use std::sync::Arc;

    const NAME: &str = "rts-0123456789ab";
    const DIGEST: &str = "docker.io/library/alpine@sha256:e311ae748be7f56ae5cba304bc488f0786df68ee507513f654a5cc8597cffb42";
    const ID: &str = "sha256:47cfb8d92856f144ddfbaa2c6ff992d155e9210b09b5213347d1a9073ea94194";

    type Calls = Arc<Mutex<Vec<(Vec<String>, Duration)>>>;
    type Reply = Box<dyn Fn(&[String]) -> RunOutput + Send + Sync>;
    struct Fake {
        calls: Calls,
        reply: Reply,
    }
    impl Runner for Fake {
        fn run(&self, argv: &[String], t: Duration) -> io::Result<RunOutput> {
            self.calls.lock().unwrap().push((argv.to_vec(), t));
            Ok((self.reply)(argv))
        }
    }
    fn fake(reply: impl Fn(&[String]) -> RunOutput + Send + Sync + 'static) -> (Box<Fake>, Calls) {
        let calls = Arc::new(Mutex::new(vec![]));
        (
            Box::new(Fake {
                calls: calls.clone(),
                reply: Box::new(reply),
            }),
            calls,
        )
    }
    fn out(code: i32, stdout: &str, stderr: &str) -> RunOutput {
        RunOutput {
            code: Some(code),
            stdout: stdout.into(),
            stderr: stderr.into(),
            killed: false,
        }
    }
    // For `docker exec <name> sh -c WRAPPER rts <nonce> …`, argv[6] is the nonce.
    fn trailer(argv: &[String], status: u32, o: u64, e: u64) -> String {
        format!("\nRTS_EXIT_{} {status} {o} {e}\n", argv[6])
    }
    fn opts() -> ExecOptions {
        ExecOptions::new(120_000, 65536)
    }
    fn spec(image: &str) -> CreateSpec<'_> {
        CreateSpec {
            name: NAME,
            image,
            memory_mib: 2048,
            cpus: 2,
            egress: None,
        }
    }

    #[test]
    fn image_references_are_content_addressed_only() {
        assert!(is_container_image(DIGEST));
        assert!(is_container_image(ID));
        assert!(!is_container_image("alpine:3.20"));
        assert!(!is_container_image("sha256:abc"));
        assert!(!is_container_image(
            "SHA256:47cfb8d92856f144ddfbaa2c6ff992d155e9210b09b5213347d1a9073ea94194"
        ));
    }

    #[test]
    fn runtime_names_are_plain_tokens() {
        assert!(is_runtime_name("sysbox-runc"));
        assert!(is_runtime_name("runc"));
        assert!(!is_runtime_name(""));
        assert!(!is_runtime_name("run c"));
        assert!(!is_runtime_name("../runc"));
        assert!(
            DockerSandbox::new(NAME, fake(|_| out(0, "", "")).0)
                .unwrap()
                .with_runtime(Some("x y".into()))
                .is_err()
        );
    }

    #[test]
    fn create_is_locked_down_and_then_started() {
        let (f, calls) = fake(|_| out(0, "", ""));
        let sb = DockerSandbox::new(NAME, f)
            .unwrap()
            .with_runtime(Some("sysbox-runc".into()))
            .unwrap();
        sb.create(&spec(DIGEST)).unwrap();
        let c = calls.lock().unwrap();
        assert_eq!(c.len(), 2);
        let a = &c[0].0;
        assert_eq!(a[0], "create");
        for flag in [
            "--network",
            "--read-only",
            "--cap-drop",
            "--security-opt",
            "--pids-limit",
            "--memory",
            "--memory-swap",
            "--cpus",
            "--tmpfs",
            "--init",
            "--runtime",
        ] {
            assert!(a.contains(&flag.to_string()), "missing {flag} in {a:?}");
        }
        let after = |flag: &str| a[a.iter().position(|x| x == flag).unwrap() + 1].clone();
        assert_eq!(after("--network"), "none");
        assert_eq!(after("--cap-drop"), "ALL");
        assert_eq!(after("--memory"), "2048m");
        assert_eq!(after("--cpus"), "2");
        assert_eq!(after("--runtime"), "sysbox-runc");
        assert_eq!(after("--tmpfs"), "/tmp:rw,exec,nosuid,size=2048m");
        let image_at = a.iter().position(|x| x == DIGEST).unwrap();
        assert_eq!(a[image_at + 1..].to_vec(), sv(&KEEPALIVE));
        assert_eq!(c[1].0, sv(&["start", NAME]));
    }

    #[test]
    fn create_without_init_omits_the_flag_and_an_image_id_is_accepted() {
        let (f, calls) = fake(|_| out(0, "", ""));
        let sb = DockerSandbox::new(NAME, f).unwrap().with_init(false);
        sb.create(&spec(ID)).unwrap();
        let a = &calls.lock().unwrap()[0].0;
        assert!(!a.contains(&"--init".to_string()));
        assert!(!a.contains(&"--runtime".to_string()));
        assert!(a.contains(&ID.to_string()));
    }

    #[test]
    fn create_refuses_tags_small_machines_and_network_sessions() {
        let (f, calls) = fake(|_| out(0, "", ""));
        let sb = DockerSandbox::new(NAME, f).unwrap();
        assert!(
            sb.create(&spec("alpine:3.20"))
                .unwrap_err()
                .to_string()
                .contains("digest")
        );
        assert!(
            sb.create(&CreateSpec {
                memory_mib: 32,
                ..spec(DIGEST)
            })
            .unwrap_err()
            .to_string()
            .contains("64 MiB")
        );
        let hosts = crate::network::test_allow_hosts(&["198.51.100.7:443"]);
        let egress = Egress {
            allow_hosts: &hosts,
            attach_log: std::path::Path::new("/dev/null"),
            session_hours: 1,
        };
        let e = sb
            .create(&CreateSpec {
                egress: Some(egress),
                ..spec(DIGEST)
            })
            .unwrap_err()
            .to_string();
        assert!(e.contains("network sessions"), "{e}");
        assert!(
            calls.lock().unwrap().is_empty(),
            "nothing may reach docker before validation passes"
        );
    }

    #[test]
    fn exec_wraps_the_command_and_reads_the_trailer() {
        let (f, calls) = fake(|a| out(0, "hi\n", &format!("warn\n{}", trailer(a, 3, 3, 5))));
        let sb = DockerSandbox::new(NAME, f).unwrap();
        let r = sb
            .exec(
                &sv(&["echo", "hi"]),
                &ExecOptions {
                    workdir: Some("/in".into()),
                    ..opts()
                },
            )
            .unwrap();
        assert_eq!(
            (r.exit_code, r.stdout.as_slice(), r.stderr.as_slice()),
            (3, &b"hi\n"[..], &b"warn\n"[..])
        );
        let (a, t) = &calls.lock().unwrap()[0];
        assert_eq!(a[..2].to_vec(), sv(&["exec", NAME]));
        assert_eq!(a[2..5].to_vec(), sv(&["sh", "-c", WRAPPER]));
        assert_eq!(a[5], "rts");
        assert_eq!(
            (a[7].as_str(), a[8].as_str(), a[9].as_str()),
            ("120", "65536", "/in")
        );
        assert_eq!(a[10..].to_vec(), sv(&["echo", "hi"]));
        assert_eq!(*t, Duration::from_secs(120) + BACKSTOP);
    }

    #[test]
    fn a_missing_trailer_marks_the_container_unavailable_for_good() {
        let (f, calls) = fake(|_| out(0, "", "no trailer here\n"));
        let sb = DockerSandbox::new(NAME, f).unwrap();
        assert!(matches!(
            sb.exec(&sv(&["true"]), &opts()),
            Err(Error::Unavailable(_))
        ));
        assert!(sb.unavailable());
        assert!(matches!(
            sb.exec(&sv(&["true"]), &opts()),
            Err(Error::Unavailable(_))
        ));
        assert_eq!(
            calls.lock().unwrap().len(),
            1,
            "a dead container gets no more execs"
        );
    }

    #[test]
    fn a_gone_container_is_reported_as_unavailable() {
        let (f, _) = fake(|_| {
            out(
                1,
                "",
                "Error response from daemon: container rts-0123456789ab is not running\n",
            )
        });
        let sb = DockerSandbox::new(NAME, f).unwrap();
        let e = sb.exec(&sv(&["true"]), &opts()).unwrap_err().to_string();
        assert!(e.contains("gone"), "{e}");
    }

    #[test]
    fn a_host_backstop_kill_is_unavailable_not_a_timeout() {
        let (f, _) = fake(|_| RunOutput {
            code: None,
            killed: true,
            ..Default::default()
        });
        let sb = DockerSandbox::new(NAME, f).unwrap();
        assert!(matches!(
            sb.exec(&sv(&["sleep", "9"]), &opts()),
            Err(Error::Unavailable(_))
        ));
    }

    #[test]
    fn remove_forces_and_verifies() {
        let (f, calls) = fake(|a| {
            if a[0] == "ps" {
                out(0, "", "")
            } else {
                out(0, "", "")
            }
        });
        let sb = DockerSandbox::new(NAME, f).unwrap();
        sb.remove().unwrap();
        let c = calls.lock().unwrap();
        assert_eq!(c[0].0, sv(&["rm", "-f", NAME]));
        assert_eq!(c[1].0[..3].to_vec(), sv(&["ps", "-a", "--filter"]));
        assert!(matches!(
            sb.exec(&sv(&["true"]), &opts()),
            Err(Error::Unavailable(_))
        ));
    }

    #[test]
    fn remove_fails_when_the_container_survives() {
        let (f, _) = fake(|a| {
            if a[0] == "ps" {
                out(0, &format!("{NAME}\n"), "")
            } else {
                out(1, "", "rm: busy")
            }
        });
        let sb = DockerSandbox::new(NAME, f).unwrap();
        let e = sb.remove().unwrap_err().to_string();
        assert!(e.contains("still present") && e.contains("busy"), "{e}");
    }

    #[test]
    fn listing_keeps_only_rts_names() {
        let (f, _) = fake(|_| out(0, "rts-0123456789ab\nother\nrts-zz\n", ""));
        assert_eq!(list_containers(&*f).unwrap(), vec![NAME.to_string()]);
    }
}
