//! `rts tool <name>`: one sandboxed tool call, spawned by the opencode
//! plugin shim. The request is one JSON object on stdin:
//!
//! ```json
//! { "options": { "machine": "rts-…", "mvmctl": "…", "emptyGitConfig": "…",
//!                "stateDir": "…", "timeoutMs": 120000, "maxOutputBytes": 65536,
//!                "cwd": "/", "mvmHome": "…", "proxy": true },
//!   "input": { …the tool's own arguments… } }
//! ```
//!
//! The tool's text goes to stdout with exit 0. A failed call puts its
//! message on stderr and exits 1, or 3 when the machine is unavailable,
//! so the shim can tell a lost VM from an ordinary tool error.

use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use serde::Deserialize;
use serde_json::Value;

use crate::docker::{DockerRunner, DockerSandbox, is_runtime_name};
use crate::mvm::{MvmSandbox, SpawnRunner, StateDir, is_machine_name};
use crate::sandbox::{Error, Exec};
use crate::tools::{SANDBOXED_TOOLS, ToolLimits, Tools};

pub const EXIT_OK: i32 = 0;
pub const EXIT_TOOL_ERROR: i32 = 1;
pub const EXIT_BAD_REQUEST: i32 = 2;
pub const EXIT_UNAVAILABLE: i32 = 3;

/// Largest request accepted on stdin. A write's content travels here.
const MAX_REQUEST_BYTES: u64 = 64 * 1024 * 1024;

/// Which sandbox a session runs in. The plugin passes the whole `bridge`
/// object through, so a session picks its backend once, here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    #[default]
    Mvm,
    Docker,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct BridgeOptions {
    pub machine: String,
    #[serde(default)]
    pub backend: Backend,
    /// mvm backend: the mvmctl binary and the empty git config it runs with.
    #[serde(default)]
    pub mvmctl: Option<PathBuf>,
    #[serde(default)]
    pub empty_git_config: Option<PathBuf>,
    /// Container backend: the docker CLI (default `docker` on PATH) and the
    /// runtime the container was created with, for error messages only.
    #[serde(default)]
    pub docker: Option<PathBuf>,
    #[serde(default)]
    pub runtime: Option<String>,
    pub state_dir: PathBuf,
    pub timeout_ms: u64,
    pub max_output_bytes: u64,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub mvm_home: Option<String>,
    /// Network session: `shell` runs under the guest proxy environment.
    #[serde(default)]
    pub proxy: bool,
    /// Host directory for the reviewer's copy of /out (`record::Record`).
    #[serde(default)]
    pub record_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub options: BridgeOptions,
    pub input: Value,
}

pub struct Outcome {
    pub code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl Outcome {
    fn ok(text: String) -> Self {
        Self {
            code: EXIT_OK,
            stdout: text,
            stderr: String::new(),
        }
    }
    fn fail(code: i32, msg: String) -> Self {
        Self {
            code,
            stdout: String::new(),
            stderr: msg,
        }
    }
}

pub fn parse_request(raw: &[u8]) -> Result<Request, String> {
    let r: Request = serde_json::from_slice(raw).map_err(|e| format!("bad tool request: {e}"))?;
    let o = &r.options;
    if !is_machine_name(&o.machine) {
        return Err(format!(
            "bad tool request: machine must match rts-<12 hex chars>, got {:?}",
            o.machine
        ));
    }
    if o.timeout_ms == 0 || o.max_output_bytes == 0 {
        return Err("bad tool request: timeoutMs and maxOutputBytes must be positive".into());
    }
    if o.record_dir.as_ref().is_some_and(|d| !d.is_absolute()) {
        return Err("bad tool request: recordDir must be an absolute host path".into());
    }
    if let Some(cwd) = &o.cwd
        && !cwd.starts_with('/')
    {
        return Err("bad tool request: cwd must be an absolute guest path".into());
    }
    match o.backend {
        Backend::Mvm if o.mvmctl.is_none() || o.empty_git_config.is_none() => {
            return Err("bad tool request: the mvm backend needs mvmctl and emptyGitConfig".into());
        }
        Backend::Docker if o.runtime.as_deref().is_some_and(|r| !is_runtime_name(r)) => {
            return Err("bad tool request: runtime must be a plain runtime name".into());
        }
        _ => {}
    }
    Ok(r)
}

/// Run one call against any `Exec`. Split out so tests can drive it
/// without mvmctl.
pub fn call(
    tool: &str,
    sb: &dyn Exec,
    o: &BridgeOptions,
    input: &Value,
    leak_shell: bool,
) -> Outcome {
    if !SANDBOXED_TOOLS.contains(&tool) {
        return Outcome::fail(
            EXIT_BAD_REQUEST,
            format!("tool not allowed in sandbox: {tool}"),
        );
    }
    let limits = ToolLimits {
        timeout_ms: o.timeout_ms,
        max_output_bytes: o.max_output_bytes,
        cwd: o.cwd.clone().unwrap_or_else(|| "/".into()),
        leak_shell,
        proxy: o.proxy,
        record: o.record_dir.clone(),
    };
    match (Tools { sb, limits }).call(tool, input) {
        Ok(text) => Outcome::ok(text),
        Err(e @ Error::Unavailable(_)) => Outcome::fail(EXIT_UNAVAILABLE, e.to_string()),
        Err(e) => Outcome::fail(EXIT_TOOL_ERROR, e.to_string()),
    }
}

/// The real entry point: read stdin, attach to the session's machine, run.
pub fn run(tool: &str) -> Outcome {
    let mut raw = vec![];
    if let Err(e) = std::io::stdin()
        .take(MAX_REQUEST_BYTES + 1)
        .read_to_end(&mut raw)
    {
        return Outcome::fail(
            EXIT_BAD_REQUEST,
            format!("reading the tool request failed: {e}"),
        );
    }
    if raw.len() as u64 > MAX_REQUEST_BYTES {
        return Outcome::fail(
            EXIT_BAD_REQUEST,
            format!("tool request is over {MAX_REQUEST_BYTES} bytes"),
        );
    }
    let req = match parse_request(&raw) {
        Ok(r) => r,
        Err(e) => return Outcome::fail(EXIT_BAD_REQUEST, e),
    };
    let o = &req.options;
    let sb: Box<dyn Exec> = match o.backend {
        Backend::Mvm => {
            let mut extra_env = HashMap::new();
            if let Some(h) = &o.mvm_home {
                extra_env.insert("HOME".to_string(), h.clone());
            }
            // parse_request checked both are present.
            let (mvmctl, empty_git_config) = (
                o.mvmctl.clone().unwrap_or_default(),
                o.empty_git_config.clone().unwrap_or_default(),
            );
            let runner = SpawnRunner {
                mvmctl,
                empty_git_config,
                extra_env,
                remove_env: Vec::new(),
            };
            match MvmSandbox::new(&o.machine, Box::new(runner)) {
                Ok(sb) => Box::new(sb.with_state(StateDir(o.state_dir.clone()))),
                Err(e) => return Outcome::fail(EXIT_BAD_REQUEST, e.to_string()),
            }
        }
        Backend::Docker => {
            let runner = DockerRunner {
                docker: o.docker.clone().unwrap_or_else(|| PathBuf::from("docker")),
            };
            match DockerSandbox::new(&o.machine, Box::new(runner))
                .and_then(|sb| sb.with_runtime(o.runtime.clone()))
            {
                Ok(sb) => Box::new(sb.with_state(StateDir(o.state_dir.clone()))),
                Err(e) => return Outcome::fail(EXIT_BAD_REQUEST, e.to_string()),
            }
        }
    };
    let leak = std::env::var("RTS_TEST_LEAK").as_deref() == Ok("shell");
    if leak {
        eprintln!("rts: RTS_TEST_LEAK=shell is set; shell runs ON THE HOST (leak check only)");
    }
    call(tool, sb.as_ref(), o, &req.input, leak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sandbox::sv;
    use crate::testutil::{Recorder, ok};
    use serde_json::json;

    fn request(extra: Value) -> Value {
        let mut o = json!({
            "machine": "rts-0123456789ab", "mvmctl": "/x/mvmctl", "emptyGitConfig": "/x/g",
            "stateDir": "/x/state", "timeoutMs": 1000, "maxOutputBytes": 100,
        });
        o.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        json!({ "options": o, "input": { "command": "true" } })
    }

    fn parse(v: Value) -> Result<Request, String> {
        parse_request(v.to_string().as_bytes())
    }

    #[test]
    fn the_backend_defaults_to_mvm_and_docker_needs_no_mvmctl() {
        assert_eq!(
            parse(request(json!({}))).unwrap().options.backend,
            Backend::Mvm
        );
        let docker = json!({ "options": { "machine": "rts-0123456789ab", "backend": "docker", "runtime": "sysbox-runc", "stateDir": "/x/state", "timeoutMs": 1000, "maxOutputBytes": 100 }, "input": {} });
        let r = parse(docker).unwrap();
        assert_eq!(r.options.backend, Backend::Docker);
        assert_eq!(r.options.runtime.as_deref(), Some("sysbox-runc"));
        assert!(r.options.mvmctl.is_none());
        let no_mvmctl = json!({ "options": { "machine": "rts-0123456789ab", "stateDir": "/x/state", "timeoutMs": 1000, "maxOutputBytes": 100 }, "input": {} });
        assert!(parse(no_mvmctl).unwrap_err().contains("mvmctl"));
        let bad_runtime = json!({ "options": { "machine": "rts-0123456789ab", "backend": "docker", "runtime": "run c", "stateDir": "/x/state", "timeoutMs": 1000, "maxOutputBytes": 100 }, "input": {} });
        assert!(parse(bad_runtime).unwrap_err().contains("runtime"));
        assert!(parse(request(json!({ "backend": "firecracker" }))).is_err());
    }

    #[test]
    fn a_well_formed_request_parses() {
        let r = parse(request(json!({ "cwd": "/in", "mvmHome": "/home/op" }))).unwrap();
        assert_eq!(r.options.cwd.as_deref(), Some("/in"));
        assert_eq!(r.input["command"], "true");
    }

    #[test]
    fn proxy_defaults_to_false_and_parses_when_set() {
        assert!(!parse(request(json!({}))).unwrap().options.proxy);
        assert!(
            parse(request(json!({ "proxy": true })))
                .unwrap()
                .options
                .proxy
        );
        assert!(
            !parse(request(json!({ "proxy": false })))
                .unwrap()
                .options
                .proxy
        );
        assert!(parse(request(json!({ "proxy": "yes" }))).is_err());
    }

    #[test]
    fn a_proxied_request_runs_shell_under_the_proxy_env() {
        let rec = Recorder::new(|_, _| ok(b""));
        let r = parse(request(json!({ "proxy": true }))).unwrap();
        let out = call(
            "shell",
            &rec,
            &r.options,
            &json!({ "command": "id" }),
            false,
        );
        assert_eq!(out.code, EXIT_OK);
        let a = &rec.argvs()[0];
        assert_eq!(a[0], "env");
        assert_eq!(a[1..9].to_vec(), crate::network::proxy_env());
        assert_eq!(a[9..].to_vec(), sv(&["sh", "-c", "id"]));
    }

    #[test]
    fn bad_requests_are_refused_before_anything_runs() {
        assert!(
            parse(request(json!({ "machine": "other-vm" })))
                .unwrap_err()
                .contains("machine")
        );
        assert!(
            parse(request(json!({ "timeoutMs": 0 })))
                .unwrap_err()
                .contains("positive")
        );
        assert!(
            parse(request(json!({ "cwd": "relative" })))
                .unwrap_err()
                .contains("absolute")
        );
        assert!(
            parse(request(json!({ "net": true })))
                .unwrap_err()
                .contains("unknown field")
        );
        assert!(parse_request(b"not json").is_err());
    }

    #[test]
    fn a_tool_outside_the_six_is_refused() {
        let rec = Recorder::new(|_, _| ok(b""));
        let r = parse(request(json!({}))).unwrap();
        let out = call("webfetch", &rec, &r.options, &json!({}), false);
        assert_eq!(out.code, EXIT_BAD_REQUEST);
        assert!(rec.argvs().is_empty());
    }

    #[test]
    fn an_unavailable_machine_gets_its_own_exit_code() {
        struct Dead;
        impl Exec for Dead {
            fn exec(
                &self,
                _: &[String],
                _: &crate::sandbox::ExecOptions,
            ) -> crate::sandbox::Result<crate::sandbox::ExecResult> {
                Err(Error::Unavailable("sandbox removed".into()))
            }
        }
        let r = parse(request(json!({}))).unwrap();
        let out = call(
            "shell",
            &Dead,
            &r.options,
            &json!({ "command": "id" }),
            false,
        );
        assert_eq!(out.code, EXIT_UNAVAILABLE);
        assert!(out.stderr.contains("sandbox removed"));
    }

    #[test]
    fn a_tool_error_is_exit_one_and_success_is_stdout() {
        let rec = Recorder::new(|_, _| ok(b"hello"));
        let r = parse(request(json!({}))).unwrap();
        let out = call(
            "shell",
            &rec,
            &r.options,
            &json!({ "command": "echo hello" }),
            false,
        );
        assert_eq!((out.code, out.stdout.as_str()), (EXIT_OK, "hello"));
        let out = call("shell", &rec, &r.options, &json!({}), false);
        assert_eq!(out.code, EXIT_TOOL_ERROR);
        assert!(out.stderr.contains("command is required"));
    }
}
