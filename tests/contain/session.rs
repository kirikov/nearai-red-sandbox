//! Containment through real opencode 2.2.2. Plugin-loading checks run
//! `opencode run --standalone --model none/none`: plugins load, then the run
//! fails at model resolution before any inference request, so no endpoint
//! or tokens are needed.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use rts::config::ProviderConfig;
use rts::files::guest_sha256;
use rts::mvm::{list_machines, run_captured};
use rts::protocol::{new_nonce, random_hex};
use rts::sandbox::{Exec, sv};
use rts::session::{SessionPaths, ancestor_config, opencode_launch, prepare_session_dirs, session_paths, write_opencode_config};
use serde_json::{Value, json};

use crate::harness::{Harness, bridge_options, fake_ironwire, fresh_machine, limits, mvm, opencode, rts_tool, tempdir};

const ALLOWED_TOOLS: [&str; 7] = ["edit", "glob", "grep", "question", "read", "shell", "write"];

/// Every tool real opencode offers before our plugin's transform runs, as
/// `opencode_offers_exactly_the_recorded_built_in_tools` measures it.
const BUILTIN_TOOLS: &[&str] = &[
    "shell", "read", "write", "edit", "glob", "grep", "question", "patch", "webfetch", "websearch", "skill", "subagent",
    "opencode_list_mcp_resources", "opencode_read_mcp_resource", "opencode_models", "opencode_session_move",
    "opencode_session_rename", "opencode_worktree_create", "opencode_worktree_list", "opencode_worktree_remove",
];

fn plugin_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("plugin")
}

fn provider() -> ProviderConfig {
    ProviderConfig { id: "ironwire".into(), base_url: "http://127.0.0.1:8463/openai/v1".into(), models: vec!["z-ai/glm-5.3-flash".into()], api_key_env: None }
}

struct Session {
    root: PathBuf,
    p: SessionPaths,
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn session() -> Session {
    let root = tempdir("e2e");
    let p = session_paths(&random_hex(6), &root.join("sessions"), &root.join("rt"));
    prepare_session_dirs(&p).unwrap();
    Session { root, p }
}

/// Plugin options as the launcher writes them.
fn options(h: &Harness, p: &SessionPaths, machine: &str, nonce: &str) -> Value {
    json!({
        "machine": machine,
        "nonce": nonce,
        "readyFile": p.ready_file,
        "auditFile": p.audit_file,
        "provider": "ironwire",
        "models": provider().models,
        "rts": env!("CARGO_BIN_EXE_rts"),
        "bridge": bridge_options(h, machine, &p.state_dir),
        "debugFile": p.runtime.join("debug.json"),
    })
}

fn configure(h: &Harness, p: &SessionPaths, machine: &str, nonce: &str) -> PathBuf {
    write_opencode_config(&p.config_dir, &plugin_dir(), &options(h, p, machine, nonce), &provider(), false, None).unwrap()
}

/// Load plugins through real opencode without spending inference.
fn probe(oc: &str, p: &SessionPaths) -> String {
    let l = opencode_launch(p, oc, std::env::vars());
    let mut cmd = Command::new(oc);
    cmd.args(["run", "--standalone", "--model", "none/none", "probe"]).current_dir(&l.cwd).env_clear().envs(l.env.iter().map(|(k, v)| (k, v)));
    let r = run_captured(cmd, Duration::from_secs(90)).unwrap();
    String::from_utf8_lossy(&r.stderr).into_owned()
}

fn ready_holds(p: &SessionPaths, nonce: &str) -> bool {
    fs::read_to_string(&p.ready_file).is_ok_and(|s| s == nonce)
}

fn edit_config(file: &Path, f: impl FnOnce(&mut Value)) {
    let mut cfg: Value = serde_json::from_str(&fs::read_to_string(file).unwrap()).unwrap();
    f(&mut cfg);
    fs::write(file, cfg.to_string()).unwrap();
}

#[test]
#[ignore = "real mvm and opencode"]
fn the_plugin_loads_reports_ready_and_leaves_exactly_the_seven_tools() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let m = fresh_machine(h);
    let s = session();
    let nonce = new_nonce();
    configure(h, &s.p, &m.name, &nonce);
    let err = probe(&oc, &s.p);
    assert!(ready_holds(&s.p, &nonce), "{err}");
    let debug: Value = serde_json::from_str(&fs::read_to_string(s.p.runtime.join("debug.json")).unwrap()).unwrap();
    assert_eq!(debug["tools"], json!(ALLOWED_TOOLS));
    assert_eq!(debug["mcp"], json!([]));
}

#[test]
#[ignore = "real mvm and opencode"]
fn a_project_opencode_json_with_an_mcp_server_is_refused_by_the_launcher_and_its_server_never_spawns() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let m = fresh_machine(h);
    let s = session();
    let sentinel = s.root.join("mcp-spawned");
    let evil = json!({ "mcp": { "evil": { "type": "local", "command": ["sh", "-c", format!("touch '{}'; sleep 5", sentinel.display())] } } });
    fs::write(s.p.work.join("opencode.json"), evil.to_string()).unwrap();
    // The launcher refuses a work dir that holds config before it starts opencode.
    assert!(ancestor_config(&s.p.work).contains(&s.p.work.join("opencode.json")));
    // And if opencode is started anyway, the plugin's MCP removal keeps the server from spawning.
    configure(h, &s.p, &m.name, &new_nonce());
    probe(&oc, &s.p);
    thread::sleep(Duration::from_secs(2));
    assert!(!sentinel.exists());
}

#[test]
#[ignore = "real mvm and opencode"]
fn a_second_plugin_stray_tool_is_removed_before_the_session_is_ready() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let m = fresh_machine(h);
    let s = session();
    let stray = s.root.join("stray-plugin");
    fs::create_dir(&stray).unwrap();
    fs::write(stray.join("package.json"), r#"{"name":"stray","version":"0.0.0","type":"module","main":"index.ts","exports":{".":"./index.ts"}}"#).unwrap();
    fs::write(
        stray.join("index.ts"),
        "export default { id: \"stray\", setup: async (ctx) => { await ctx.tool.transform((e) => e.add({ name: \"stray\", description: \"stray\", input: { type: \"object\", properties: {} }, execute: async () => ({ content: \"stray\" }) })) } }\n",
    )
    .unwrap();
    let nonce = new_nonce();
    let file = configure(h, &s.p, &m.name, &nonce);
    // Loads before ours, so the sandbox transform removes its tool.
    edit_config(&file, |c| c["plugin"].as_array_mut().unwrap().insert(0, json!([stray, {}])));
    let err = probe(&oc, &s.p);
    assert!(ready_holds(&s.p, &nonce), "{err}");
    let debug: Value = serde_json::from_str(&fs::read_to_string(s.p.runtime.join("debug.json")).unwrap()).unwrap();
    assert_eq!(debug["tools"], json!(ALLOWED_TOOLS));
}

#[test]
#[ignore = "real mvm and opencode"]
fn the_plugin_configured_as_a_file_path_is_skipped_so_no_ready_file_appears() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let m = fresh_machine(h);
    let s = session();
    let nonce = new_nonce();
    let file = configure(h, &s.p, &m.name, &nonce);
    edit_config(&file, |c| c["plugin"][0][0] = json!(plugin_dir().join("index.ts")));
    probe(&oc, &s.p);
    assert!(!ready_holds(&s.p, &nonce));
}

/// A plugin at `dir` with `index.ts` holding `source`, loaded by its directory.
fn local_plugin(dir: &Path, name: &str, source: &str) {
    fs::create_dir(dir).unwrap();
    fs::write(dir.join("package.json"), json!({ "name": name, "version": "0.0.0", "type": "module", "main": "index.ts", "exports": { ".": "./index.ts" } }).to_string()).unwrap();
    fs::write(dir.join("index.ts"), source).unwrap();
}

#[test]
#[ignore = "real mvm and opencode"]
fn opencode_offers_exactly_the_recorded_built_in_tools() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let m = fresh_machine(h);
    let s = session();
    let out = s.root.join("tools.json");
    let inventory = s.root.join("inventory-plugin");
    // Loads before ours, so it sees every tool opencode offers.
    local_plugin(
        &inventory,
        "inventory",
        &format!(
            "import {{ writeFileSync }} from \"node:fs\"\nexport default {{ id: \"inventory\", setup: async (ctx) => {{ await ctx.tool.transform((e) => {{ writeFileSync({}, JSON.stringify(e.list().map((t) => t.id ?? t.name))) }}) }} }}\n",
            json!(out)
        ),
    );
    let nonce = new_nonce();
    let file = configure(h, &s.p, &m.name, &nonce);
    edit_config(&file, |c| c["plugin"].as_array_mut().unwrap().insert(0, json!([inventory, {}])));
    let err = probe(&oc, &s.p);
    let mut seen: Vec<String> = serde_json::from_str(&fs::read_to_string(&out).unwrap_or_else(|e| panic!("no tools.json ({e}): {err}"))).unwrap();
    seen.sort();
    let mut want: Vec<String> = BUILTIN_TOOLS.iter().map(|t| t.to_string()).collect();
    want.sort();
    assert_eq!(seen, want, "{err}");
    assert!(ready_holds(&s.p, &nonce), "{err}");
}

#[test]
#[ignore = "real mvm and opencode"]
fn an_mcp_server_in_the_global_config_never_spawns() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let m = fresh_machine(h);
    let s = session();
    let sentinel = s.root.join("mcp-global");
    let nonce = new_nonce();
    let file = configure(h, &s.p, &m.name, &nonce);
    edit_config(&file, |c| {
        c["mcp"] = json!({ "evil": { "type": "local", "command": ["sh", "-c", format!("touch '{}'; sleep 5", sentinel.display())] } })
    });
    let err = probe(&oc, &s.p);
    thread::sleep(Duration::from_secs(2));
    assert!(!sentinel.exists());
    assert!(ready_holds(&s.p, &nonce), "{err}");
    let debug: Value = serde_json::from_str(&fs::read_to_string(s.p.runtime.join("debug.json")).unwrap()).unwrap();
    assert_eq!(debug["mcp"], json!([]));
}

/// The leak check reruns this test with RTS_TEST_LEAK=shell and requires
/// it to fail on the host-sentinel line, so keep that check first.
#[test]
#[ignore = "real mvm"]
fn sentinel_write_and_shell_touch_land_in_the_guest_and_not_on_the_host() {
    let Some(h) = mvm() else { return };
    let m = fresh_machine(h);
    let s = session();
    let id = random_hex(6);
    let (s1, s2) = (format!("/tmp/rts-sentinel-{id}"), format!("/tmp/rts-sentinel2-{id}"));
    let o = bridge_options(h, &m.name, &s.p.state_dir);
    let w = rts_tool(&o, "write", json!({ "path": s1, "content": "sentinel" }));
    let t = rts_tool(&o, "shell", json!({ "command": format!("touch {s2}") }));
    let leaked: Vec<&String> = [&s1, &s2].into_iter().filter(|p| Path::new(p).exists()).collect();
    for p in [&s1, &s2] {
        let _ = fs::remove_file(p);
    }
    assert!(leaked.is_empty(), "LEAK: host sentinel present: {leaked:?}");
    assert_eq!((w.0, t.0), (0, 0), "{w:?} {t:?}");
    assert_eq!(guest_sha256(&*m, &s1).unwrap().len(), 64);
    assert_eq!(m.exec(&sv(&["test", "-e", &s2]), &limits()).unwrap().exit_code, 0);
}

#[test]
#[ignore = "real mvm"]
fn lost_vm_after_machine_stop_tool_calls_fail_and_nothing_runs_on_the_host() {
    let Some(h) = mvm() else { return };
    let m = fresh_machine(h);
    let s = session();
    let o = bridge_options(h, &m.name, &s.p.state_dir);
    assert_eq!(rts_tool(&o, "shell", json!({ "command": "echo up" })).1, "up");
    let stopped = h.run.run(&sv(&["machine", "stop", &m.name, "--yes"]), Duration::from_secs(120)).unwrap();
    assert_eq!(stopped.code, Some(0), "{}", String::from_utf8_lossy(&stopped.stderr));
    let listed = h.run.run(&sv(&["machine", "ps", "--json"]), Duration::from_secs(30)).unwrap();
    assert_eq!(listed.code, Some(0), "{}", String::from_utf8_lossy(&listed.stderr));
    let inventory: Value = serde_json::from_slice(&listed.stdout).unwrap();
    let machine = inventory.as_array().unwrap().iter().find(|v| v["name"] == m.name).expect("stopped persistent machine is still listed");
    assert_eq!(machine["status"], "stopped");
    let sentinel = format!("/tmp/rts-sentinel3-{}", random_hex(6));
    let a = rts_tool(&o, "shell", json!({ "command": format!("touch {sentinel}") }));
    let b = rts_tool(&o, "write", json!({ "path": sentinel, "content": "x" }));
    assert_eq!((a.0, b.0), (3, 3), "{a:?} {b:?}");
    assert!(a.2.contains("sandbox unavailable"));
    assert!(!Path::new(&sentinel).exists());
}

#[test]
#[ignore = "real mvm and opencode"]
fn sigint_to_the_launcher_exports_out_and_leaves_no_rts_machine() {
    let (Some(h), Some(oc)) = (mvm(), opencode()) else { return };
    let root = tempdir("e2e");
    let cfg = root.join("session.json");
    let model = "z-ai/glm-5.3-flash";
    let config = json!({
        "provider": { "baseURL": fake_ironwire(model), "models": [model] },
        "sessionsDir": root.join("sessions"),
        "mvmctl": h.mvmctl,
        "opencode": oc,
        "limits": { "handshakeMs": 60_000 },
    });
    fs::write(&cfg, config.to_string()).unwrap();
    // A fake external IronWire is not attested: allow it explicitly.
    let mut child = Command::new(env!("CARGO_BIN_EXE_rts"))
        .args(["launch", "--allow-unattested-provider", "--config"])
        .arg(&cfg)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let err = Arc::new(Mutex::new(String::new()));
    let (e, mut stderr) = (err.clone(), child.stderr.take().unwrap());
    let reader = thread::spawn(move || {
        let mut buf = [0u8; 4096];
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            e.lock().unwrap().push_str(&String::from_utf8_lossy(&buf[..n]));
        }
    });
    let deadline = Instant::now() + Duration::from_secs(120);
    while !err.lock().unwrap().contains("plugin ready") && Instant::now() < deadline && child.try_wait().unwrap().is_none() {
        thread::sleep(Duration::from_millis(200));
    }
    let text = err.lock().unwrap().clone();
    assert!(text.contains("plugin ready"), "{text}");
    let machine = text.split("created ").nth(1).unwrap()[..16].to_string();
    let o = bridge_options(h, &machine, &root.join("state"));
    assert_eq!(rts_tool(&o, "write", json!({ "path": "/out/hello.txt", "content": "hi" })).0, 0);
    rts::sys::send_signal(child.id() as i32, rts::sys::SIGINT);
    child.wait().unwrap();
    reader.join().unwrap();
    let text = err.lock().unwrap().clone();
    assert!(text.contains("teardown"), "{text}");
    assert!(!list_machines(h.run.as_ref()).unwrap().contains(&machine));
    let id = &machine[4..];
    assert_eq!(fs::read_to_string(root.join("sessions").join(id).join("out/hello.txt")).unwrap(), "hi");
    fs::remove_dir_all(root).unwrap();
}
