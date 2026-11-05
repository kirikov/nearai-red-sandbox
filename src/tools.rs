//! The six sandboxed tools. Every effect goes through `Exec`; nothing here
//! touches the host filesystem or spawns a host process, except the
//! explicit leak seam used only by the leak-detection test, and the host's
//! copy of what the reviewer writes under /out (`record`).

use std::path::{Component, Path, PathBuf};

use serde_json::Value;

use crate::files::{GuestRead, guest_read, guest_write, script};
use crate::glob::glob_match;
use crate::record::{Record, rel_under_out};
use crate::sandbox::{Error, Exec, ExecOptions, Result, sv};

pub const SANDBOXED_TOOLS: [&str; 6] = ["shell", "read", "write", "edit", "glob", "grep"];

#[derive(Debug, Clone)]
pub struct ToolLimits {
    pub timeout_ms: u64,
    pub max_output_bytes: u64,
    /// Guest directory that relative paths and shell commands start from.
    pub cwd: String,
    /// TEST ONLY: run `shell` on the host so the leak check can prove the
    /// containment suite notices. Never set in a real session.
    pub leak_shell: bool,
    /// Network session: run `shell` under the guest proxy environment
    /// (`network::proxy_env`), which `machine exec` does not inherit.
    pub proxy: bool,
    /// Host directory that keeps the reviewer's own copy of every `write`
    /// and `edit` under /out (`record::Record`); None records nothing.
    pub record: Option<PathBuf>,
}

const READ_LINE_CAP: usize = 2000;
const SEARCH_LIMIT: usize = 100;
const LISTING_CAP: u64 = 512 * 1024; // stays under mvm's silent ~950 KB stdout truncation

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

pub fn count_matches(text: &str, needle: &str) -> usize {
    if needle.is_empty() { 0 } else { text.matches(needle).count() }
}

pub fn apply_edit(text: &str, old: &str, new: &str, replace_all: bool) -> Result<String> {
    match count_matches(text, old) {
        0 => Err("oldString not found".into()),
        n if n > 1 && !replace_all => Err(Error::Other(format!("oldString matches {n} times; set replaceAll"))),
        _ if replace_all => Ok(text.replace(old, new)),
        _ => Ok(text.replacen(old, new, 1)),
    }
}

fn join_output(parts: Vec<String>) -> String {
    parts.into_iter().filter(|p| !p.is_empty()).collect::<Vec<_>>().join("\n")
}

fn str_field<'a>(input: &'a Value, k: &str) -> Option<&'a str> {
    input.get(k).and_then(Value::as_str)
}

fn positive(input: &Value, k: &str, fallback: usize) -> usize {
    match input.get(k).and_then(Value::as_f64) {
        Some(v) if v.is_finite() && v >= 1.0 => v as usize,
        _ => fallback,
    }
}

fn is_true(input: &Value, k: &str) -> bool {
    input.get(k).and_then(Value::as_bool) == Some(true)
}

/// Lexical POSIX path resolution against the guest cwd (no filesystem access).
pub fn resolve(cwd: &str, p: &str) -> String {
    let joined = if p.starts_with('/') { PathBuf::from(p) } else { Path::new(cwd).join(p) };
    let mut out: Vec<String> = vec![];
    for c in joined.components() {
        match c {
            Component::Normal(s) => out.push(s.to_string_lossy().into_owned()),
            Component::ParentDir => {
                out.pop();
            }
            _ => {}
        }
    }
    format!("/{}", out.join("/"))
}

fn is_hidden(rel: &str) -> bool {
    rel.split('/').any(|s| s.starts_with('.'))
}

pub struct Tools<'a> {
    pub sb: &'a dyn Exec,
    pub limits: ToolLimits,
}

impl Tools<'_> {
    fn opts(&self) -> ExecOptions {
        ExecOptions::new(self.limits.timeout_ms, self.limits.max_output_bytes)
    }

    fn path(&self, input: &Value, k: &str) -> Result<String> {
        match str_field(input, k) {
            Some(p) if !p.is_empty() => Ok(resolve(&self.limits.cwd, p)),
            _ => Err(Error::Other(format!("{k} is required"))),
        }
    }

    pub fn call(&self, tool: &str, input: &Value) -> Result<String> {
        match tool {
            "shell" => self.shell(input),
            "read" => self.read(input),
            "write" => self.write(input),
            "edit" => self.edit(input),
            "glob" => self.glob(input),
            "grep" => self.grep(input),
            other => Err(Error::Other(format!("tool not allowed in sandbox: {other}"))),
        }
    }

    pub fn shell(&self, input: &Value) -> Result<String> {
        let command = str_field(input, "command").ok_or("command is required")?;
        if is_true(input, "background") {
            return Err("background commands are not supported in the sandbox".into());
        }
        if self.limits.leak_shell {
            let o = std::process::Command::new("sh").args(["-c", command]).output().map_err(|e| Error::Other(e.to_string()))?;
            let code = o.status.code().unwrap_or(-1);
            return Ok(join_output(vec![text(&o.stdout).trim_end().into(), text(&o.stderr).trim_end().into(), if code != 0 { format!("exit status: {code}") } else { String::new() }]));
        }
        let requested = input.get("timeout").and_then(Value::as_f64).filter(|t| *t > 0.0).map_or(self.limits.timeout_ms, |t| t as u64);
        let timeout_ms = requested.min(self.limits.timeout_ms);
        let workdir = match str_field(input, "workdir") {
            Some(w) => resolve(&self.limits.cwd, w),
            None => self.limits.cwd.clone(),
        };
        let mut argv = if self.limits.proxy { std::iter::once("env".to_string()).chain(crate::network::proxy_env()).collect() } else { vec![] };
        argv.extend(sv(&["sh", "-c", command]));
        let r = self.sb.exec(&argv, &ExecOptions { timeout_ms, workdir: Some(workdir), ..self.opts() })?;
        let mut parts = vec![text(&r.stdout).trim_end().to_string(), text(&r.stderr).trim_end().to_string()];
        if r.stdout_bytes > r.stdout.len() as u64 {
            parts.push(format!("[output truncated: {} bytes total]", r.stdout_bytes));
        }
        if r.stderr_bytes > r.stderr.len() as u64 {
            parts.push(format!("[stderr truncated: {} bytes total]", r.stderr_bytes));
        }
        if r.timed_out {
            parts.push(format!("[timed out after {}s]", timeout_ms.div_ceil(1000)));
        } else if r.exit_code != 0 {
            parts.push(format!("exit status: {}", r.exit_code));
        }
        let out = join_output(parts);
        Ok(if out.is_empty() { "(no output)".into() } else { out })
    }

    pub fn read(&self, input: &Value) -> Result<String> {
        let path = self.path(input, "path")?;
        let offset = positive(input, "offset", 1);
        let limit = positive(input, "limit", READ_LINE_CAP).min(READ_LINE_CAP);
        match guest_read(self.sb, &path, self.limits.max_output_bytes)? {
            GuestRead::Missing => Err(Error::Other(format!("not found: {path}"))),
            GuestRead::Dir { entries, truncated } => {
                let shown: Vec<String> = entries.iter().skip(offset - 1).take(limit).cloned().collect();
                let mut parts = vec![shown.join("\n")];
                if offset - 1 + limit < entries.len() {
                    parts.push(format!("({} entries; use offset to see more)", entries.len()));
                }
                if truncated {
                    parts.push("(listing truncated)".into());
                }
                let out = join_output(parts);
                Ok(if out.is_empty() { "(empty directory)".into() } else { out })
            }
            GuestRead::File { data, size, truncated } => {
                // Never fails on bad bytes: invalid UTF-8 becomes U+FFFD.
                let t = text(&data);
                if t.is_empty() {
                    return Ok(if truncated { format!("[file truncated: 0 of {size} bytes shown]") } else { "(empty file)".into() });
                }
                let mut lines: Vec<&str> = t.split('\n').collect();
                if lines.last() == Some(&"") {
                    lines.pop();
                }
                let shown: Vec<String> = lines.iter().enumerate().skip(offset - 1).take(limit).map(|(i, l)| format!("{}: {l}", i + 1)).collect();
                let mut parts = vec![shown.join("\n")];
                if offset - 1 + limit < lines.len() {
                    parts.push(format!("({} lines; use offset to read more)", lines.len()));
                }
                if truncated {
                    parts.push(format!("[file truncated: showing the first {} of {size} bytes]", data.len()));
                }
                Ok(join_output(parts))
            }
        }
    }

    /// The record and the path under /out, when this path is recorded.
    fn recorded<'p>(&self, path: &'p str) -> Option<(Record, &'p str)> {
        let dir = self.limits.record.as_ref()?;
        Some((Record::new(dir), rel_under_out(path)?))
    }

    fn keep(&self, path: &str, data: &[u8], guest_base: bool) -> Result<()> {
        if let Some((rec, rel)) = self.recorded(path) {
            rec.put(rel, data, guest_base).map_err(|e| Error::Other(format!("recording {path} on the host failed: {e}")))?;
        }
        Ok(())
    }

    pub fn write(&self, input: &Value) -> Result<String> {
        let path = self.path(input, "path")?;
        let content = str_field(input, "content").ok_or("content is required")?;
        self.keep(&path, content.as_bytes(), false)?;
        guest_write(self.sb, &path, content.as_bytes())?;
        Ok(format!("wrote {} bytes to {path}", content.len()))
    }

    pub fn edit(&self, input: &Value) -> Result<String> {
        let path = self.path(input, "path")?;
        let (Some(old), Some(new)) = (str_field(input, "oldString"), str_field(input, "newString")) else {
            return Err("oldString and newString are required".into());
        };
        if old == new {
            return Err("oldString and newString are identical".into());
        }
        let replace_all = is_true(input, "replaceAll");
        // A file the reviewer wrote is edited from the host's copy: the
        // guest's copy may have been rewritten since.
        if let Some(t) = self.recorded(&path).and_then(|(rec, rel)| rec.get(rel)) {
            if old.is_empty() {
                return Err("oldString is empty; use write to replace a whole file".into());
            }
            let t = String::from_utf8(t).map_err(|_| Error::Other("file is not valid UTF-8; edit refused".into()))?;
            let n = if replace_all { count_matches(&t, old) } else { 1 };
            let updated = apply_edit(&t, old, new, replace_all)?;
            self.keep(&path, updated.as_bytes(), false)?;
            guest_write(self.sb, &path, updated.as_bytes())?;
            return Ok(format!("edited {path}: {n} replacement{}", if n == 1 { "" } else { "s" }));
        }
        match guest_read(self.sb, &path, self.limits.max_output_bytes)? {
            GuestRead::Missing if old.is_empty() => {
                self.keep(&path, new.as_bytes(), false)?;
                guest_write(self.sb, &path, new.as_bytes())?;
                Ok(format!("created {path}"))
            }
            GuestRead::Missing => Err(Error::Other(format!("not found: {path}"))),
            GuestRead::Dir { .. } => Err(Error::Other(format!("is a directory: {path}"))),
            GuestRead::File { truncated: true, .. } => Err(Error::Other(format!("file larger than {} bytes; edit refused", self.limits.max_output_bytes))),
            GuestRead::File { .. } if old.is_empty() => Err("oldString is empty; use write to replace a whole file".into()),
            GuestRead::File { data, .. } => {
                let t = String::from_utf8(data).map_err(|_| Error::Other("file is not valid UTF-8; edit refused".into()))?;
                let n = if replace_all { count_matches(&t, old) } else { 1 };
                let updated = apply_edit(&t, old, new, replace_all)?;
                self.keep(&path, updated.as_bytes(), true)?;
                guest_write(self.sb, &path, updated.as_bytes())?;
                Ok(format!("edited {path}: {n} replacement{}", if n == 1 { "" } else { "s" }))
            }
        }
    }

    pub fn glob(&self, input: &Value) -> Result<String> {
        let pattern = str_field(input, "pattern").filter(|p| !p.is_empty()).ok_or("pattern is required")?;
        let root = if input.get("path").is_some() { self.path(input, "path")? } else { self.limits.cwd.clone() };
        let limit = positive(input, "limit", SEARCH_LIMIT);
        let body = r#"cd -- "$1" || exit 2; find . -xdev -type f -print0 2>/dev/null; exit 0"#;
        let r = self.sb.exec(&script(body, &[&root]), &ExecOptions { max_output_bytes: LISTING_CAP, ..self.opts() })?;
        if r.exit_code != 0 {
            let e = text(&r.stderr);
            return Err(Error::Other(format!("glob {root}: {}", if e.trim().is_empty() { format!("exit status {}", r.exit_code) } else { e.trim().into() })));
        }
        let listing = text(&r.stdout);
        let mut names: Vec<&str> = listing.split('\0').collect();
        names.pop(); // "" after the final NUL, or a cut-off name
        let hidden = is_true(input, "hidden");
        let mut matches: Vec<&str> = names
            .into_iter()
            .map(|n| n.strip_prefix("./").unwrap_or(n))
            .filter(|rel| !rel.is_empty() && (hidden || !is_hidden(rel)) && glob_match(pattern, rel))
            .collect();
        matches.sort();
        if matches.is_empty() {
            return Ok(if r.truncated { "no files found (listing truncated)".into() } else { "no files found".into() });
        }
        let shown: Vec<String> = matches.iter().take(limit).map(|rel| resolve(&root, rel)).collect();
        let mut parts = vec![shown.join("\n")];
        if matches.len() > limit {
            parts.push(format!("(results truncated: showing first {limit} of {})", matches.len()));
        }
        if r.truncated {
            parts.push("(file listing truncated; narrow the path)".into());
        }
        Ok(join_output(parts))
    }

    pub fn grep(&self, input: &Value) -> Result<String> {
        let pattern = str_field(input, "pattern").filter(|p| !p.is_empty()).ok_or("pattern is required")?;
        let root = if input.get("path").is_some() { self.path(input, "path")? } else { self.limits.cwd.clone() };
        let limit = positive(input, "limit", SEARCH_LIMIT);
        let mut argv = sv(&["grep", "-rn"]);
        if is_true(input, "literal") {
            argv.push("-F".into());
        }
        if input.get("caseSensitive").and_then(Value::as_bool) == Some(false) {
            argv.push("-i".into());
        }
        if let Some(inc) = str_field(input, "include").filter(|s| !s.is_empty()) {
            argv.push(format!("--include={inc}"));
        }
        argv.extend(["--".into(), pattern.into(), root]);
        let r = self.sb.exec(&argv, &self.opts())?;
        if r.exit_code == 1 {
            return Ok("no matches".into());
        }
        let out = text(&r.stdout);
        if r.exit_code != 0 && out.is_empty() {
            let e = text(&r.stderr);
            return Err(Error::Other(format!("grep: {}", if e.trim().is_empty() { format!("exit status {}", r.exit_code) } else { e.trim().into() })));
        }
        let mut lines: Vec<&str> = out.split('\n').collect();
        if lines.last() == Some(&"") {
            lines.pop();
        }
        if r.truncated {
            lines.pop(); // the last line may be cut
        }
        let mut parts = vec![lines.iter().take(limit).copied().collect::<Vec<_>>().join("\n")];
        if lines.len() > limit {
            parts.push(format!("(results truncated: showing first {limit} matches)"));
        } else if r.truncated {
            parts.push("(output truncated; narrow the search)".into());
        }
        if r.exit_code != 0 {
            parts.push(format!("(grep reported errors: {})", text(&r.stderr).trim().chars().take(500).collect::<String>()));
        }
        let o = join_output(parts);
        Ok(if o.is_empty() { "no matches".into() } else { o })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::ExecResult;
    use crate::testutil::{Recorder, ok, reply};
    use serde_json::json;

    fn lim() -> ToolLimits {
        ToolLimits { timeout_ms: 120_000, max_output_bytes: 65536, cwd: "/".into(), leak_shell: false, proxy: false, record: None }
    }
    fn file_reply(data: &[u8], size: Option<u64>) -> ExecResult {
        let mut out = format!("F {}\n", size.unwrap_or(data.len() as u64)).into_bytes();
        out.extend_from_slice(data);
        ok(&out)
    }
    fn is_read(a: &[String]) -> bool {
        a[2].contains("printf 'D")
    }
    fn is_write_step(a: &[String]) -> bool {
        a[2].contains("base64 -d") || a[2].contains("mv -f")
    }
    fn run<F: Fn(&[String], &ExecOptions) -> ExecResult>(rec: &Recorder<F>, l: ToolLimits, tool: &str, input: Value) -> Result<String> {
        Tools { sb: rec, limits: l }.call(tool, &input)
    }

    #[test]
    fn shell_reports_nonzero_exit() {
        let rec = Recorder::new(|_, _| reply(3, b"partial\n", b""));
        let out = run(&rec, lim(), "shell", json!({"command": "make && false"})).unwrap();
        assert_eq!(rec.argvs()[0], sv(&["sh", "-c", "make && false"]));
        assert!(out.contains("partial") && out.ends_with("exit status: 3"));
    }

    #[test]
    fn shell_in_a_network_session_runs_under_the_proxy_env() {
        let rec = Recorder::new(|_, _| ok(b""));
        run(&rec, ToolLimits { proxy: true, ..lim() }, "shell", json!({"command": "pip download six"})).unwrap();
        assert_eq!(
            rec.argvs()[0],
            sv(&[
                "env",
                "HTTPS_PROXY=http://127.0.0.1:1080",
                "HTTP_PROXY=http://127.0.0.1:1080",
                "ALL_PROXY=socks5h://127.0.0.1:1080",
                "NO_PROXY=localhost,127.0.0.1,::1",
                "https_proxy=http://127.0.0.1:1080",
                "http_proxy=http://127.0.0.1:1080",
                "all_proxy=socks5h://127.0.0.1:1080",
                "no_proxy=localhost,127.0.0.1,::1",
                "sh",
                "-c",
                "pip download six",
            ])
        );
        let rec = Recorder::new(|_, _| ok(b""));
        run(&rec, ToolLimits { proxy: false, ..lim() }, "shell", json!({"command": "pip download six"})).unwrap();
        assert_eq!(rec.argvs()[0], sv(&["sh", "-c", "pip download six"]));
    }

    #[test]
    fn shell_workdir_resolves_against_the_guest_cwd() {
        let rec = Recorder::new(|_, _| ok(b""));
        run(&rec, lim(), "shell", json!({"command": "ls", "workdir": "src"})).unwrap();
        assert_eq!(rec.execs.lock().unwrap()[0].1.workdir.as_deref(), Some("/src"));
    }

    #[test]
    fn shell_timeout_and_truncation_markers() {
        let rec = Recorder::new(|_, _| reply(137, b"", b""));
        assert!(run(&rec, lim(), "shell", json!({"command": "sleep 999"})).unwrap().ends_with("[timed out after 120s]"));
        let rec = Recorder::new(|_, _| ExecResult { stdout_bytes: 2097152, truncated: true, ..ok(b"xxxxxxxxxx") });
        assert!(run(&rec, lim(), "shell", json!({"command": "yes"})).unwrap().ends_with("[output truncated: 2097152 bytes total]"));
    }

    #[test]
    fn shell_timeout_defaults_and_caps() {
        let rec = Recorder::new(|_, _| ok(b""));
        for t in [json!(0), Value::Null, json!(5000), json!(10_000_000)] {
            run(&rec, lim(), "shell", json!({"command": "x", "timeout": t})).unwrap();
        }
        let ts: Vec<u64> = rec.execs.lock().unwrap().iter().map(|(_, o)| o.timeout_ms).collect();
        assert_eq!(ts, vec![120_000, 120_000, 5000, 120_000]);
    }

    #[test]
    fn shell_refuses_background_and_includes_stderr() {
        let rec = Recorder::new(|_, _| ok(b""));
        let e = run(&rec, lim(), "shell", json!({"command": "serve", "background": true})).unwrap_err();
        assert_eq!(e.to_string(), "background commands are not supported in the sandbox");
        assert!(rec.argvs().is_empty());
        let rec = Recorder::new(|_, _| reply(0, b"out\n", b"warn\n"));
        assert_eq!(run(&rec, lim(), "shell", json!({"command": "x"})).unwrap(), "out\nwarn");
    }

    #[test]
    fn read_numbers_lines_and_pages() {
        let rec = Recorder::new(|_, _| file_reply(b"a\nb\nc\nd\n", None));
        assert_eq!(run(&rec, lim(), "read", json!({"path": "/f"})).unwrap(), "1: a\n2: b\n3: c\n4: d");
        assert_eq!(run(&rec, lim(), "read", json!({"path": "/f", "offset": 2, "limit": 2})).unwrap(), "2: b\n3: c\n(4 lines; use offset to read more)");
        assert_eq!(&rec.argvs()[0][3..], &sv(&["rts", "/f", "65536"])[..]);
    }

    #[test]
    fn read_caps_at_2000_lines() {
        let t: String = (1..=2500).map(|i| format!("l{i}")).collect::<Vec<_>>().join("\n");
        let rec = Recorder::new(move |_, _| file_reply(t.as_bytes(), None));
        let out = run(&rec, lim(), "read", json!({"path": "/f", "limit": 5000})).unwrap();
        assert_eq!(out.lines().filter(|l| l.contains(": l")).count(), 2000);
        assert!(out.contains("2000: l2000"));
    }

    #[test]
    fn read_directory_missing_and_invalid_utf8() {
        let rec = Recorder::new(|_, _| ok(b"D\nb.txt\0a/\0"));
        assert_eq!(run(&rec, lim(), "read", json!({"path": "/d"})).unwrap(), "a/\nb.txt");
        let rec = Recorder::new(|_, _| ok(b"M\n"));
        assert_eq!(run(&rec, lim(), "read", json!({"path": "/nope"})).unwrap_err().to_string(), "not found: /nope");
        let rec = Recorder::new(|_, _| file_reply(&[0x68, 0x69, 0xff, 0xfe, 0x0a, 0xc3], None));
        assert_eq!(run(&rec, lim(), "read", json!({"path": "/bin"})).unwrap(), "1: hi\u{FFFD}\u{FFFD}\n2: \u{FFFD}");
    }

    #[test]
    fn read_notes_truncation_and_resolves_relative_paths() {
        let rec = Recorder::new(|_, _| file_reply(b"abc", Some(999_999)));
        assert!(run(&rec, lim(), "read", json!({"path": "/big"})).unwrap().contains("999999"));
        let rec = Recorder::new(|_, _| ok(b"M\n"));
        let _ = run(&rec, ToolLimits { cwd: "/in".into(), ..lim() }, "read", json!({"path": "notes.txt"}));
        assert_eq!(rec.argvs()[0][4], "/in/notes.txt");
        let _ = run(&rec, ToolLimits { cwd: "/in".into(), ..lim() }, "read", json!({"path": "../etc/x"}));
        assert_eq!(rec.argvs()[1][4], "/etc/x");
    }

    #[test]
    fn write_sends_content() {
        let rec = Recorder::new(|_, _| ok(b""));
        assert!(run(&rec, lim(), "write", json!({"path": "/out/a.txt", "content": "héllo"})).unwrap().contains("/out/a.txt"));
        use base64::Engine;
        let a = rec.argvs();
        assert_eq!(base64::engine::general_purpose::STANDARD.decode(a[0].last().unwrap()).unwrap(), "héllo".as_bytes());
        assert_eq!(a.last().unwrap().last().unwrap(), "/out/a.txt");
    }

    #[test]
    fn writes_under_out_are_recorded_on_the_host_and_others_are_not() {
        let t = tempfile::tempdir().unwrap();
        let rec = Recorder::new(|_, _| ok(b""));
        let l = ToolLimits { record: Some(t.path().to_path_buf()), ..lim() };
        run(&rec, l.clone(), "write", json!({"path": "/out/verdict.json", "content": "{}"})).unwrap();
        run(&rec, l, "write", json!({"path": "/tmp/x", "content": "y"})).unwrap();
        let r = crate::record::Record::new(t.path());
        assert_eq!(r.get("verdict.json").unwrap(), b"{}");
        assert_eq!(r.list(), vec!["verdict.json".to_string()]);
    }

    #[test]
    fn an_edit_of_a_recorded_file_ignores_the_guest_copy() {
        let t = tempfile::tempdir().unwrap();
        let r = crate::record::Record::new(t.path());
        r.put("findings.md", b"high: F-01", false).unwrap();
        // The guest rewrote its copy; the edit must not start from it.
        let rec = Recorder::new(|a, _| if is_read(a) { file_reply(b"no findings", None) } else { ok(b"") });
        let l = ToolLimits { record: Some(t.path().to_path_buf()), ..lim() };
        run(&rec, l, "edit", json!({"path": "/out/findings.md", "oldString": "high", "newString": "critical"})).unwrap();
        assert_eq!(r.get("findings.md").unwrap(), b"critical: F-01");
        assert!(rec.argvs().iter().all(|a| !is_read(a)));
    }

    #[test]
    fn an_edit_of_a_guest_produced_file_is_recorded_as_such() {
        let t = tempfile::tempdir().unwrap();
        let rec = Recorder::new(|a, _| if is_read(a) { file_reply(b"log a", None) } else { ok(b"") });
        let l = ToolLimits { record: Some(t.path().to_path_buf()), ..lim() };
        run(&rec, l, "edit", json!({"path": "/out/log.txt", "oldString": "a", "newString": "b"})).unwrap();
        let out = t.path().join("export");
        let doc = crate::record::overlay(&crate::record::Record::new(t.path()), &out).unwrap();
        assert_eq!(doc["files"]["log.txt"]["source"], "reviewer-edit-of-guest-file");
    }

    #[test]
    fn apply_edit_semantics() {
        assert_eq!(apply_edit("a foo b", "foo", "bar", false).unwrap(), "a bar b");
        assert_eq!(apply_edit("cost $1", "$1", "$&$2", false).unwrap(), "cost $&$2");
        assert_eq!(apply_edit("abc", "x", "y", false).unwrap_err().to_string(), "oldString not found");
        assert_eq!(apply_edit("x x", "x", "y", false).unwrap_err().to_string(), "oldString matches 2 times; set replaceAll");
        assert_eq!(apply_edit("x x x", "x", "y", true).unwrap(), "y y y");
    }

    #[test]
    fn edit_writes_back_the_replacement() {
        let rec = Recorder::new(|a, _| if is_read(a) { file_reply(b"one two\n", None) } else { ok(b"") });
        assert!(run(&rec, lim(), "edit", json!({"path": "/w/f", "oldString": "two", "newString": "2"})).unwrap().contains("1 replacement"));
        use base64::Engine;
        let chunk = rec.argvs().into_iter().find(|a| a[2].contains("base64 -d")).unwrap();
        assert_eq!(base64::engine::general_purpose::STANDARD.decode(chunk.last().unwrap()).unwrap(), b"one 2\n");
    }

    #[test]
    fn edit_refuses_truncated_and_non_utf8_without_writing() {
        let rec = Recorder::new(|a, _| if is_read(a) { file_reply(&[b'x'; 100], Some(70_000)) } else { ok(b"") });
        let e = run(&rec, lim(), "edit", json!({"path": "/big", "oldString": "x", "newString": "y", "replaceAll": true})).unwrap_err();
        assert_eq!(e.to_string(), "file larger than 65536 bytes; edit refused");
        assert!(!rec.argvs().iter().any(|a| is_write_step(a)));
        let rec = Recorder::new(|a, _| if is_read(a) { file_reply(&[0x61, 0xff], None) } else { ok(b"") });
        assert!(run(&rec, lim(), "edit", json!({"path": "/bin", "oldString": "a", "newString": "b"})).unwrap_err().to_string().contains("not valid UTF-8"));
        assert!(!rec.argvs().iter().any(|a| is_write_step(a)));
    }

    #[test]
    fn edit_missing_file_and_identical_strings() {
        let rec = Recorder::new(|a, _| if is_read(a) { ok(b"M\n") } else { ok(b"") });
        assert_eq!(run(&rec, lim(), "edit", json!({"path": "/nope", "oldString": "a", "newString": "b"})).unwrap_err().to_string(), "not found: /nope");
        run(&rec, lim(), "edit", json!({"path": "/new", "oldString": "", "newString": "fresh"})).unwrap();
        assert!(rec.argvs().iter().any(|a| is_write_step(a)));
        assert!(run(&rec, lim(), "edit", json!({"path": "/f", "oldString": "a", "newString": "a"})).unwrap_err().to_string().contains("identical"));
    }

    fn listing(names: &[&str]) -> Vec<u8> {
        names.iter().flat_map(|n| format!("./{n}\0").into_bytes()).collect()
    }

    #[test]
    fn glob_matches_the_guest_listing() {
        let l = listing(&["a.ts", "src/b.ts", "src/c.js", "deep/x/y.ts"]);
        let rec = Recorder::new(move |_, _| ok(&l));
        let out = run(&rec, lim(), "glob", json!({"pattern": "**/*.ts", "path": "/repo"})).unwrap();
        assert_eq!(out, "/repo/a.ts\n/repo/deep/x/y.ts\n/repo/src/b.ts");
        assert_eq!(&rec.argvs()[0][3..], &sv(&["rts", "/repo"])[..]);
    }

    #[test]
    fn glob_hidden_limit_and_empty() {
        let l = listing(&["a.ts", ".git/h.ts", "src/.cache/z.ts", ".env.ts"]);
        let rec = Recorder::new(move |_, _| ok(&l));
        assert_eq!(run(&rec, lim(), "glob", json!({"pattern": "**/*.ts", "path": "/r"})).unwrap(), "/r/a.ts");
        assert_eq!(run(&rec, lim(), "glob", json!({"pattern": "**/*.ts", "path": "/r", "hidden": true})).unwrap().lines().count(), 4);
        let l = listing(&["a.ts", "b.ts", "c.ts"]);
        let rec = Recorder::new(move |_, _| ok(&l));
        let out = run(&rec, lim(), "glob", json!({"pattern": "*.ts", "path": "/r", "limit": 2})).unwrap();
        assert!(out.starts_with("/r/a.ts\n/r/b.ts\n") && out.contains("truncated"));
        assert_eq!(run(&rec, lim(), "glob", json!({"pattern": "*.rs", "path": "/r"})).unwrap(), "no files found");
    }

    #[test]
    fn grep_flags_and_double_dash() {
        let rec = Recorder::new(|_, _| ok(b"/r/a:1:hit\n"));
        let out = run(&rec, lim(), "grep", json!({"pattern": "-e x", "path": "/r", "include": "*.ts", "literal": true, "caseSensitive": false})).unwrap();
        assert_eq!(rec.argvs()[0], sv(&["grep", "-rn", "-F", "-i", "--include=*.ts", "--", "-e x", "/r"]));
        assert_eq!(out, "/r/a:1:hit");
        run(&rec, lim(), "grep", json!({"pattern": "a.b"})).unwrap();
        assert_eq!(rec.argvs()[1], sv(&["grep", "-rn", "--", "a.b", "/"]));
    }

    #[test]
    fn grep_no_matches_errors_and_limit() {
        let rec = Recorder::new(|_, _| reply(1, b"", b""));
        assert_eq!(run(&rec, lim(), "grep", json!({"pattern": "zzz"})).unwrap(), "no matches");
        let rec = Recorder::new(|_, _| reply(2, b"", b"grep: bad regex"));
        assert!(run(&rec, lim(), "grep", json!({"pattern": "("})).unwrap_err().to_string().contains("bad regex"));
        let rec = Recorder::new(|_, _| ok(b"1\n2\n3\n"));
        let out = run(&rec, lim(), "grep", json!({"pattern": "x", "limit": 2})).unwrap();
        assert!(out.starts_with("1\n2\n") && out.contains("truncated"));
    }

    #[test]
    fn unknown_tool_is_refused() {
        let rec = Recorder::new(|_, _| ok(b""));
        assert_eq!(run(&rec, lim(), "webfetch", json!({})).unwrap_err().to_string(), "tool not allowed in sandbox: webfetch");
        assert!(rec.argvs().is_empty());
    }

    #[test]
    fn leak_seam_is_off_by_default_and_runs_on_host_when_on() {
        let rec = Recorder::new(|_, _| ok(b""));
        run(&rec, lim(), "shell", json!({"command": "true"})).unwrap();
        assert_eq!(rec.argvs().len(), 1);
        let rec = Recorder::new(|_, _| ok(b""));
        let out = run(&rec, ToolLimits { leak_shell: true, ..lim() }, "shell", json!({"command": "echo on-host"})).unwrap();
        assert!(out.contains("on-host"));
        assert!(rec.argvs().is_empty());
    }

    #[test]
    fn resolve_is_lexical() {
        assert_eq!(resolve("/", "a/b"), "/a/b");
        assert_eq!(resolve("/in", "../x"), "/x");
        assert_eq!(resolve("/in", "/abs/./y"), "/abs/y");
        assert_eq!(resolve("/", "-rf"), "/-rf");
        assert_eq!(resolve("/", "../../.."), "/");
    }
}
