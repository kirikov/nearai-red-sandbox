//! `rts run`: one headless opencode session on the container backend, for
//! a Linux host without a hypervisor of its own (a confidential VM under
//! dstack). The launcher creates the container, copies the inputs in, runs
//! `opencode run` with the prompt until it exits or the run limit passes,
//! exports /out and removes the container. No terminal, no mvm preflight,
//! no splash: every line goes to launcher.log and, plainly, to stderr.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::config::OperatorConfig;
use crate::copy::CopyLimits;
use crate::docker::{DockerRunner, DockerSandbox, is_container_image, list_containers};
use crate::mvm::{Runner, StateDir};
use crate::protocol::{new_nonce, random_hex};
use crate::sandbox::{CopyOutResult, CreateSpec, Sandbox};
use crate::session::{
    Deps, SessionPaths, ancestor_config, check_opencode, check_provider, initialize_guest_storage,
    new_session_id, opencode_launch_with_key, prepare_session_dirs, session_paths, wait_for_ready,
    write_opencode_config,
};
use crate::sys;

/// Options of `rts run`, from its flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOptions {
    /// The guest image: a repo digest or a local image id.
    pub image: String,
    /// Copied to /in before opencode starts.
    pub inputs: Option<PathBuf>,
    /// The file holding the prompt opencode runs.
    pub prompt_file: PathBuf,
    /// The docker CLI; None is `docker` on PATH.
    pub docker: Option<PathBuf>,
    /// `--runtime` for the container; None is the daemon's default.
    pub runtime: Option<String>,
    /// The whole run, from container creation to opencode's exit.
    pub timeout_min: u64,
}

pub const DEFAULT_TIMEOUT_MIN: u64 = 60;
/// A prompt is one message; anything bigger belongs in /in.
pub const MAX_PROMPT_BYTES: u64 = 256 * 1024;

/// What `opencode run` is given, besides the prompt.
pub fn opencode_run_argv(opencode: &str, model: &str, prompt: &str) -> Vec<String> {
    [
        opencode,
        "run",
        "--standalone",
        "--format",
        "json",
        "--auto",
        "--model",
        model,
        prompt,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// The plugin options for a container-backed session: the bridge names the
/// docker backend and carries no mvm fields.
pub fn plugin_options(
    p: &SessionPaths,
    cfg: &OperatorConfig,
    o: &RunOptions,
    rts: &Path,
    nonce: &str,
) -> Value {
    let mut bridge = json!({
        "machine": p.machine,
        "backend": "docker",
        "stateDir": p.state_dir,
        "timeoutMs": cfg.limits.tool_timeout_ms,
        "maxOutputBytes": cfg.limits.max_output_bytes,
        "recordDir": record_dir(p),
    });
    if let Some(d) = &o.docker {
        bridge["docker"] = json!(d);
    }
    if let Some(r) = &o.runtime {
        bridge["runtime"] = json!(r);
    }
    json!({
        "machine": p.machine,
        "nonce": nonce,
        "readyFile": p.ready_file,
        "auditFile": p.audit_file,
        "provider": cfg.provider.id,
        "models": cfg.provider.models,
        "rts": rts,
        "bridge": bridge,
    })
}

/// Where the host keeps the reviewer's copy of /out: the runtime dir, never
/// inside the guest and never in the results until the export.
pub fn record_dir(p: &SessionPaths) -> PathBuf {
    p.runtime.join("record")
}

/// Read the prompt: UTF-8, non-blank, at most [`MAX_PROMPT_BYTES`].
pub fn read_prompt(file: &Path) -> Result<String, String> {
    let meta = fs::metadata(file).map_err(|e| format!("prompt file {}: {e}", file.display()))?;
    if meta.len() > MAX_PROMPT_BYTES {
        return Err(format!(
            "prompt file {} is over {MAX_PROMPT_BYTES} bytes; put material in the inputs directory instead",
            file.display()
        ));
    }
    let text =
        fs::read_to_string(file).map_err(|e| format!("prompt file {}: {e}", file.display()))?;
    if text.trim().is_empty() {
        return Err(format!("prompt file {} is empty", file.display()));
    }
    Ok(text)
}

struct Logger {
    file: PathBuf,
}

impl Logger {
    fn log(&self, m: &str) {
        eprintln!("[rts] {m}");
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if let Ok(mut f) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file)
        {
            let _ = writeln!(f, "{secs} {m}");
        }
    }
}

struct Live<'a> {
    log: &'a Logger,
    sb: &'a DockerSandbox,
    paths: &'a SessionPaths,
    /// `docker create` was attempted; a container may exist.
    attempted: bool,
    /// `docker create` and `start` succeeded; /out is worth exporting.
    created: bool,
    child: Option<Child>,
    opencode_exit: Option<i32>,
    /// out-provenance.json, once the recorded files replaced the export.
    provenance: Option<Value>,
}

impl Live<'_> {
    fn teardown(&mut self, why: &str) -> Option<CopyOutResult> {
        self.log.log(&format!("teardown: {why}"));
        if let Some(mut c) = self.child.take() {
            sys::send_signal(c.id() as i32, sys::SIGTERM);
            let end = Instant::now() + Duration::from_secs(10);
            while c.try_wait().ok().flatten().is_none() && Instant::now() < end {
                thread::sleep(Duration::from_millis(100));
            }
            let _ = c.kill();
            let _ = c.wait();
            self.log.log("teardown: opencode stopped");
        }
        if !self.attempted {
            return None;
        }
        let mut exported = None;
        if self.created {
            let state = StateDir(self.paths.state_dir.clone());
            let quiesced = state.quiesce(Duration::from_secs(30));
            if let Err(e) = &quiesced {
                self.log.log(&format!(
                    "teardown: EXPORT SKIPPED: cannot quiesce tools: {e}"
                ));
            }
            if quiesced.is_ok() {
                match self.sb.copy_out("/out", &self.paths.out_dir) {
                    Ok(r) => {
                        match &r {
                            CopyOutResult::Exported { files, bytes } => self.log.log(&format!(
                                "teardown: exported {files} files, {bytes} bytes to {}",
                                self.paths.out_dir.display()
                            )),
                            CopyOutResult::NothingToExport => {
                                self.log.log("teardown: nothing to export")
                            }
                        }
                        exported = Some(r);
                    }
                    Err(e) => self.log.log(&format!("teardown: EXPORT FAILED: {e}")),
                }
            }
        }
        // The reviewer's own writes win over the guest's copy, exported or not.
        if self.created {
            let rec = crate::record::Record::new(record_dir(self.paths));
            match crate::record::overlay(&rec, &self.paths.out_dir) {
                Ok(doc) => {
                    let differed = doc["guestCopyDiffered"].as_array().map_or(0, Vec::len);
                    let n = doc["files"].as_object().map_or(0, |m| m.values().filter(|v| v["source"] != "guest").count());
                    self.log.log(&format!(
                        "teardown: {n} files from the reviewer's recorded writes; the guest's copy differed for {differed}"
                    ));
                    let path = self.paths.session_dir.join("out-provenance.json");
                    if let Err(e) = fs::write(&path, serde_json::to_string_pretty(&doc).unwrap() + "\n") {
                        self.log.log(&format!("teardown: cannot write {}: {e}", path.display()));
                    }
                    self.provenance = Some(doc);
                }
                Err(e) => self.log.log(&format!("teardown: RECORDED OUT FAILED: {e}")),
            }
        }
        // A failed create may still have left a container: remove either way.
        match self.sb.remove() {
            Ok(()) => self
                .log
                .log(&format!("teardown: removed {}", self.paths.machine)),
            Err(e) => self
                .log
                .log(&format!("teardown: CONTAINER REMOVAL FAILED: {e}")),
        }
        exported
    }
}

/// Run one headless session. Returns opencode's exit code, or 1 when the
/// run was refused, failed or hit its limit.
pub fn run_headless(cfg: &OperatorConfig, o: &RunOptions, deps: Deps) -> i32 {
    let id = deps.session_id.clone().unwrap_or_else(new_session_id);
    let runtime = std::env::temp_dir().join(format!("rts-{id}-{}", random_hex(4)));
    let p = session_paths(&id, &cfg.sessions_dir, &runtime);
    let log = Logger {
        file: p.log_file.clone(),
    };
    if let Err(e) = prepare_session_dirs(&p) {
        log.log(&format!("cannot create session directories: {e}"));
        return 1;
    }
    log.log(&format!(
        "session {id}: results in {}, runtime in {}",
        p.session_dir.display(),
        p.runtime.display()
    ));
    let runner: Arc<dyn Runner> = Arc::new(DockerRunner {
        docker: o.docker.clone().unwrap_or_else(|| PathBuf::from("docker")),
    });
    // The launcher's handle does not share the state dir's marker: teardown
    // marks every tool process's view unavailable first, then still exports.
    let sb = match DockerSandbox::new(&p.machine, Box::new(runner.clone()))
        .and_then(|sb| sb.with_runtime(o.runtime.clone()))
    {
        Ok(sb) => sb
            .with_copy_limits(CopyLimits::default())
            .with_start_timeout(Duration::from_millis(cfg.limits.start_timeout_ms)),
        Err(e) => {
            log.log(&e.to_string());
            return 1;
        }
    };
    sys::catch_termination_signals();
    let started = Instant::now();
    let mut live = Live {
        log: &log,
        sb: &sb,
        paths: &p,
        attempted: false,
        created: false,
        child: None,
        opencode_exit: None,
        provenance: None,
    };
    let outcome = body(cfg, o, &deps, &runner, &mut live);
    let (code, reason) = match &outcome {
        Ok(code) => (*code, "opencode exited".to_string()),
        Err(e) => {
            log.log(e);
            (1, e.clone())
        }
    };
    StateDir(p.state_dir.clone()).mark("session ended");
    let exported = live.teardown("session ended");
    let record = json!({
        "backend": "docker",
        "image": o.image,
        "runtime": o.runtime,
        "model": format!("{}/{}", cfg.provider.id, cfg.provider.models[0]),
        "attested": cfg.embedded_inference,
        "inference": crate::config::inference_mode(&cfg.provider.id),
        "opencodeExit": live.opencode_exit,
        "exit": code,
        "reason": reason,
        "durationSeconds": started.elapsed().as_secs(),
        "exported": exported.map(|r| match r {
            CopyOutResult::Exported { files, bytes } => json!({ "files": files, "bytes": bytes }),
            CopyOutResult::NothingToExport => json!({ "files": 0, "bytes": 0 }),
        }),
        "outProvenance": live.provenance.as_ref().map(|d| json!({
            "file": "out-provenance.json",
            "guestCopyDiffered": d["guestCopyDiffered"],
        })),
    });
    if let Err(e) = fs::write(
        &p.session_record,
        serde_json::to_string_pretty(&record).unwrap() + "\n",
    ) {
        log.log(&format!("cannot write {}: {e}", p.session_record.display()));
    }
    code
}

fn interrupted(log: &Logger) -> Option<String> {
    sys::take_signal().map(|s| {
        let name = if s == sys::SIGINT {
            "SIGINT"
        } else {
            "SIGTERM"
        };
        log.log(&format!("received {name}"));
        format!("interrupted by {name}")
    })
}

fn body(
    cfg: &OperatorConfig,
    o: &RunOptions,
    deps: &Deps,
    runner: &Arc<dyn Runner>,
    live: &mut Live,
) -> Result<i32, String> {
    let p = live.paths;
    if deps.env.iter().any(|(k, _)| k == "RTS_TEST_LEAK") {
        return Err("RTS_TEST_LEAK is set; refusing to start a real session".into());
    }
    if !is_container_image(&o.image) {
        return Err(format!(
            "image must be a repo digest (name@sha256:…) or an image id (sha256:…), got {:?}",
            o.image
        ));
    }
    let prompt = read_prompt(&o.prompt_file)?;
    check_opencode(deps.host.as_ref(), &cfg.opencode)?;
    let found = ancestor_config(&p.work);
    if !found.is_empty() {
        return Err(format!(
            "opencode would load config from: {}",
            found
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    check_provider(&cfg.provider)?;
    let docker_version = runner
        .run(
            &crate::sandbox::sv(&["version", "--format", "{{.Server.Version}}"]),
            Duration::from_secs(30),
        )
        .map_err(|e| format!("docker could not run: {e}"))?;
    if docker_version.code != Some(0) {
        return Err(format!(
            "docker is not usable: {}",
            crate::mvm::tail(&docker_version.stderr)
        ));
    }
    live.log.log(&format!(
        "docker daemon {}",
        String::from_utf8_lossy(&docker_version.stdout).trim()
    ));
    for name in list_containers(runner.as_ref()).map_err(|e| e.to_string())? {
        live.log.log(&format!("preserving container {name}: session ownership and liveness are unverified; inspect it before manual cleanup"));
    }
    if let Some(why) = interrupted(live.log) {
        return Err(why);
    }
    let deadline = Instant::now() + Duration::from_secs(o.timeout_min * 60);
    let fatal = deps.fatal.clone();
    // The run ends on the time limit, a signal, or the attested forwarder's
    // fatal reason (re-attestation failed): inference is gone, so stop.
    let limit = || {
        (Instant::now() >= deadline)
            .then(|| format!("run-limit: {} minutes passed", o.timeout_min))
            .or_else(|| crate::session::stop_reason(fatal.as_deref()))
    };

    live.attempted = true;
    live.sb
        .create(&CreateSpec {
            name: &p.machine,
            image: &o.image,
            memory_mib: cfg.limits.memory_mib,
            cpus: cfg.limits.cpus,
            egress: None,
        })
        .map_err(|e| e.to_string())?;
    live.created = true;
    live.log.log(&format!(
        "created {} · {} MiB · {} vCPU · runtime {} · no network",
        p.machine,
        cfg.limits.memory_mib,
        cfg.limits.cpus,
        o.runtime.as_deref().unwrap_or("default")
    ));
    initialize_guest_storage(live.sb)?;
    if let Some(inputs) = &o.inputs {
        live.sb.copy_in(inputs, "/in").map_err(|e| e.to_string())?;
        live.log.log(&format!("copied {} to /in", inputs.display()));
    }
    if let Some(why) = interrupted(live.log).or_else(limit) {
        return Err(format!("{why} before opencode started"));
    }

    let nonce = new_nonce();
    let options = plugin_options(p, cfg, o, &deps.rts, &nonce);
    write_opencode_config(
        &p.config_dir,
        &cfg.plugin_dir,
        &options,
        &cfg.provider,
        cfg.embedded_inference,
        cfg.limits.context_tokens,
    )?;
    if fs::read_dir(&p.work)
        .map_err(|e| e.to_string())?
        .next()
        .is_some()
    {
        return Err(format!("work directory is not empty: {}", p.work.display()));
    }
    let launch = opencode_launch_with_key(
        p,
        &cfg.opencode,
        deps.env.iter().cloned(),
        cfg.provider.api_key_env.as_deref(),
    );
    let model = format!("{}/{}", cfg.provider.id, cfg.provider.models[0]);
    let argv = opencode_run_argv(&cfg.opencode, &model, &prompt);
    let stdout = File::create(p.session_dir.join("opencode.jsonl"))
        .map_err(|e| format!("opencode output file: {e}"))?;
    let stderr = File::create(p.session_dir.join("opencode.log"))
        .map_err(|e| format!("opencode log file: {e}"))?;
    let mut command = Command::new(&argv[0]);
    command
        .args(&argv[1..])
        .current_dir(&launch.cwd)
        .env_clear()
        .envs(launch.env.iter().map(|(k, v)| (k, v)))
        .stdin(Stdio::null())
        .stdout(stdout)
        .stderr(stderr);
    let child = command
        .spawn()
        .map_err(|e| format!("starting {} failed: {e}", cfg.opencode))?;
    let child = live.child.insert(child);
    live.log.log(&format!(
        "opencode started (pid {}), model {model}",
        child.id()
    ));

    let mut signalled = None;
    wait_for_ready(
        &p.ready_file,
        &nonce,
        Duration::from_millis(cfg.limits.handshake_ms),
        Duration::from_millis(10),
        &mut || {
            if signalled.is_none() {
                signalled = interrupted(live.log).or_else(limit);
            }
            signalled.is_some() || child.try_wait().ok().flatten().is_some()
        },
    )
    .map_err(|e| signalled.unwrap_or_else(|| format!("refusing the session: {e}")))?;
    live.log.log("plugin ready; session running");

    loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            let code = status.code().unwrap_or(1);
            live.opencode_exit = Some(code);
            live.log.log(&format!("opencode exited {code}"));
            return Ok(code);
        }
        if let Some(why) = interrupted(live.log).or_else(limit) {
            return Err(why);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Defaults, parse_config};
    use crate::testutil::tempdir;

    fn cfg() -> OperatorConfig {
        parse_config(
            r#"{ "provider": { "id": "nearai", "baseURL": "https://cloud-api.near.ai/v1", "models": ["m"], "apiKeyEnv": "NEARAI_API_KEY" } }"#,
            Path::new("/cfg"),
            &Defaults { mvmctl: "/bin/mvmctl".into(), plugin_dir: "/repo/plugin".into(), opencode: "opencode".into(), sessions_dir: "/repo/sessions".into() },
        )
        .unwrap()
    }

    fn options() -> RunOptions {
        RunOptions {
            image: "sha256:".to_string() + &"a".repeat(64),
            inputs: None,
            prompt_file: "/p".into(),
            docker: None,
            runtime: None,
            timeout_min: 5,
        }
    }

    #[test]
    fn the_bridge_names_the_docker_backend_and_carries_no_mvm_fields() {
        let p = session_paths("0123456789ab", Path::new("/s"), Path::new("/r"));
        let v = plugin_options(
            &p,
            &cfg(),
            &options(),
            Path::new("/bin/rts"),
            "n".repeat(24).as_str(),
        );
        assert_eq!(v["bridge"]["backend"], "docker");
        assert_eq!(v["bridge"]["machine"], "rts-0123456789ab");
        assert!(v["bridge"].get("mvmctl").is_none());
        assert!(v["bridge"].get("emptyGitConfig").is_none());
        assert!(v["bridge"].get("docker").is_none());
        assert!(v["bridge"].get("runtime").is_none());
        let with = RunOptions {
            docker: Some("/usr/bin/docker".into()),
            runtime: Some("sysbox-runc".into()),
            ..options()
        };
        let v = plugin_options(
            &p,
            &cfg(),
            &with,
            Path::new("/bin/rts"),
            "n".repeat(24).as_str(),
        );
        assert_eq!(v["bridge"]["docker"], "/usr/bin/docker");
        assert_eq!(v["bridge"]["runtime"], "sysbox-runc");
        // Exactly what `rts tool` accepts for the docker backend.
        let req = json!({ "options": v["bridge"], "input": {} });
        let parsed =
            crate::bridge::parse_request(serde_json::to_string(&req).unwrap().as_bytes()).unwrap();
        assert!(matches!(
            parsed.options.backend,
            crate::bridge::Backend::Docker
        ));
    }

    #[test]
    fn opencode_runs_headless_with_the_prompt_last() {
        let argv = opencode_run_argv("/oc", "nearai/m", "review /in");
        assert_eq!(argv[0], "/oc");
        assert_eq!(argv[1], "run");
        assert!(argv.contains(&"--standalone".to_string()));
        assert!(argv.contains(&"--auto".to_string()));
        assert_eq!(argv[argv.len() - 2..], ["nearai/m", "review /in"]);
    }

    #[test]
    fn the_prompt_must_be_present_and_small() {
        let d = tempdir("prompt");
        let f = d.join("p.md");
        assert!(read_prompt(&f).unwrap_err().contains("prompt file"));
        fs::write(&f, "  \n").unwrap();
        assert!(read_prompt(&f).unwrap_err().contains("empty"));
        fs::write(&f, "review it").unwrap();
        assert_eq!(read_prompt(&f).unwrap(), "review it");
        fs::write(&f, vec![b'x'; MAX_PROMPT_BYTES as usize + 1]).unwrap();
        assert!(read_prompt(&f).unwrap_err().contains("over"));
    }
}
