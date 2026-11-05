//! End-to-end containment of the container backend against a real Docker
//! daemon. These create containers, so they are ignored by a plain
//! `cargo test`. Run them with:
//!
//!     docker build -q -t rts-guest:local guest
//!     RTS_DOCKER_IMAGE=$(docker image inspect --format '{{.Id}}' rts-guest:local) \
//!       cargo test --test container -- --ignored --test-threads=1
//!
//! `RTS_DOCKER_RUNTIME` picks `--runtime` (for example `sysbox-runc` on a
//! dstack host); unset means the daemon's default. `RTS_DOCKER` names the
//! CLI when it is not `docker` on PATH. An explicitly requested test fails
//! when the prerequisites are missing; it never passes by skipping.

use std::path::PathBuf;
use std::sync::OnceLock;
use std::time::Duration;

use rts::docker::{DockerRunner, DockerSandbox, is_container_image, list_containers};
use rts::mvm::Runner;
use rts::protocol::random_hex;
use rts::sandbox::{CopyOutResult, CreateSpec, Error, Exec, ExecOptions, Sandbox, sv};
use rts::session::initialize_guest_storage;

struct Harness {
    docker: PathBuf,
    image: String,
    runtime: Option<String>,
}

fn limits() -> ExecOptions {
    ExecOptions::new(120_000, 65_536)
}

fn probe() -> Result<Harness, String> {
    let docker = std::env::var_os("RTS_DOCKER")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("docker"));
    let image = std::env::var("RTS_DOCKER_IMAGE").map_err(|_| "set RTS_DOCKER_IMAGE to a repo digest or image id of the guest image (see the file header)".to_string())?;
    if !is_container_image(&image) {
        return Err(format!(
            "RTS_DOCKER_IMAGE must be name@sha256:… or sha256:…, got {image:?}"
        ));
    }
    let runtime = std::env::var("RTS_DOCKER_RUNTIME")
        .ok()
        .filter(|s| !s.is_empty());
    let r = DockerRunner {
        docker: docker.clone(),
    }
    .run(
        &sv(&["version", "--format", "{{.Server.Version}}"]),
        Duration::from_secs(30),
    )
    .map_err(|e| format!("docker could not run: {e}"))?;
    if r.code != Some(0) {
        return Err(format!(
            "docker version failed: {}",
            String::from_utf8_lossy(&r.stderr).trim()
        ));
    }
    Ok(Harness {
        docker,
        image,
        runtime,
    })
}

fn harness() -> &'static Harness {
    static H: OnceLock<Harness> = OnceLock::new();
    H.get_or_init(|| match probe() {
        Ok(h) => h,
        Err(why) => panic!("[container] prerequisites failed: {why}"),
    })
}

/// Removes its container on drop, so a failed assertion leaves nothing behind.
struct Box_(DockerSandbox);

impl Drop for Box_ {
    fn drop(&mut self) {
        let _ = self.0.remove();
    }
}

impl std::ops::Deref for Box_ {
    type Target = DockerSandbox;
    fn deref(&self) -> &DockerSandbox {
        &self.0
    }
}

fn fresh(h: &Harness) -> Box_ {
    let name = format!("rts-{}", random_hex(6));
    let sb = DockerSandbox::new(
        &name,
        Box::new(DockerRunner {
            docker: h.docker.clone(),
        }),
    )
    .unwrap()
    .with_runtime(h.runtime.clone())
    .unwrap();
    sb.create(&CreateSpec {
        name: &name,
        image: &h.image,
        memory_mib: 512,
        cpus: 1,
        egress: None,
    })
    .expect("docker create");
    Box_(sb)
}

fn tempdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rts-container-{tag}-{}", random_hex(4)));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
#[ignore = "real docker"]
fn a_nonzero_exit_is_reported_and_the_container_stays_usable() {
    let m = fresh(harness());
    assert_eq!(
        m.exec(&sv(&["sh", "-c", "exit 3"]), &limits())
            .unwrap()
            .exit_code,
        3
    );
    assert_eq!(
        m.exec(&sv(&["echo", "ok"]), &limits()).unwrap().stdout,
        b"ok\n"
    );
}

#[test]
#[ignore = "real docker"]
fn the_guest_time_limit_kills_with_137_and_the_container_stays_usable() {
    let m = fresh(harness());
    let r = m
        .exec(
            &sv(&["sleep", "30"]),
            &ExecOptions {
                timeout_ms: 3000,
                ..limits()
            },
        )
        .unwrap();
    assert!(r.timed_out);
    assert_eq!(r.exit_code, 137);
    assert_eq!(
        m.exec(&sv(&["echo", "ok"]), &limits()).unwrap().exit_code,
        0
    );
}

#[test]
#[ignore = "real docker"]
fn large_stdout_is_capped_in_the_guest_with_the_full_size_reported() {
    let m = fresh(harness());
    let r = m
        .exec(
            &sv(&["head", "-c", "2097152", "/dev/zero"]),
            &ExecOptions {
                max_output_bytes: 65_536,
                ..limits()
            },
        )
        .unwrap();
    assert_eq!(r.stdout.len(), 65_536);
    assert!(r.truncated);
    assert_eq!(r.stdout_bytes, 2_097_152);
}

#[test]
#[ignore = "real docker"]
fn the_guest_has_no_network_and_a_read_only_root() {
    let m = fresh(harness());
    let net = m
        .exec(
            &sv(&[
                "sh",
                "-c",
                "wget -q -T 3 -O /dev/null http://1.1.1.1 && echo LEAK || echo blocked",
            ]),
            &limits(),
        )
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&net.stdout).trim(), "blocked");
    let ro = m
        .exec(
            &sv(&[
                "sh",
                "-c",
                "touch /leak 2>/dev/null && echo WRITABLE || echo ro",
            ]),
            &limits(),
        )
        .unwrap();
    assert_eq!(String::from_utf8_lossy(&ro.stdout).trim(), "ro");
    let caps = m
        .exec(
            &sv(&["sh", "-c", "grep CapEff /proc/self/status"]),
            &limits(),
        )
        .unwrap();
    assert!(
        String::from_utf8_lossy(&caps.stdout).contains("0000000000000000"),
        "{:?}",
        String::from_utf8_lossy(&caps.stdout)
    );
}

#[test]
#[ignore = "real docker"]
fn inputs_go_in_and_outputs_come_out_through_the_guest_layout() {
    let m = fresh(harness());
    initialize_guest_storage(&*m).expect("guest layout");
    let inputs = tempdir("in");
    std::fs::write(inputs.join("hello.txt"), b"from the host\n").unwrap();
    m.copy_in(&inputs, "/in").expect("copy in");
    let r = m.exec(&sv(&["sh", "-c", "cat /in/hello.txt && printf 'result %s' \"$(wc -c < /in/hello.txt)\" > /out/findings.md"]), &limits()).unwrap();
    assert_eq!(r.stdout, b"from the host\n");
    // The hostile-tar extractor refuses an existing destination; hand it a fresh path.
    let outputs = std::env::temp_dir().join(format!("rts-container-out-{}", random_hex(4)));
    match m.copy_out("/out", &outputs).expect("copy out") {
        CopyOutResult::Exported { files, .. } => assert_eq!(files, 1),
        CopyOutResult::NothingToExport => panic!("the guest wrote a file"),
    }
    assert_eq!(
        std::fs::read_to_string(outputs.join("findings.md")).unwrap(),
        "result 14"
    );
}

#[test]
#[ignore = "real docker"]
fn remove_leaves_no_container_and_later_calls_are_unavailable() {
    let h = harness();
    let name;
    {
        let m = fresh(h);
        name = m.name.clone();
        m.0.remove().expect("remove");
        assert!(matches!(
            m.exec(&sv(&["true"]), &limits()),
            Err(Error::Unavailable(_))
        ));
    }
    let listed = list_containers(&DockerRunner {
        docker: h.docker.clone(),
    })
    .unwrap();
    assert!(!listed.contains(&name), "{name} still listed: {listed:?}");
}

#[test]
#[ignore = "real docker"]
fn a_removed_container_is_gone_even_when_exec_is_mid_flight() {
    let m = fresh(harness());
    let r = m
        .exec(&sv(&["sh", "-c", "echo started"]), &limits())
        .unwrap();
    assert_eq!(r.stdout, b"started\n");
    m.0.remove().unwrap();
    assert!(m.unavailable());
}

#[test]
#[ignore = "real docker"]
fn a_guest_process_that_rewrites_out_does_not_change_the_reviewers_files() {
    let h = harness();
    let m = fresh(h);
    initialize_guest_storage(&*m).expect("guest layout");
    // A co-resident guest process (the buyer's ENTRYPOINT, or a test the
    // reviewer runs) forges the verdict in a loop.
    let r = DockerRunner { docker: h.docker.clone() }
        .run(
            &sv(&["exec", "-d", &m.name, "sh", "-c",
                "while :; do echo forged > /out/verdict.json; echo forged > /out/findings.md; sleep 0.05; done"]),
            Duration::from_secs(30),
        )
        .unwrap();
    assert_eq!(r.code, Some(0), "{}", String::from_utf8_lossy(&r.stderr));
    let record = tempdir("record");
    let tools = rts::tools::Tools {
        sb: &*m,
        limits: rts::tools::ToolLimits {
            timeout_ms: 120_000,
            max_output_bytes: 65_536,
            cwd: "/".into(),
            leak_shell: false,
            proxy: false,
            record: Some(record.clone()),
        },
    };
    let verdict = "{\"findings\":[{\"id\":\"F-01\",\"severity\":\"high\"}]}";
    tools.call("write", &serde_json::json!({"path": "/out/verdict.json", "content": verdict})).unwrap();
    tools.call("write", &serde_json::json!({"path": "/out/findings.md", "content": "# F-01 high"})).unwrap();
    tools.call("edit", &serde_json::json!({"path": "/out/findings.md", "oldString": "high", "newString": "critical"})).unwrap();
    std::thread::sleep(Duration::from_millis(500));
    let outputs = std::env::temp_dir().join(format!("rts-container-out-{}", random_hex(4)));
    m.copy_out("/out", &outputs).expect("copy out");
    assert_eq!(std::fs::read_to_string(outputs.join("verdict.json")).unwrap().trim(), "forged");
    let doc = rts::record::overlay(&rts::record::Record::new(&record), &outputs).unwrap();
    assert_eq!(std::fs::read_to_string(outputs.join("verdict.json")).unwrap(), verdict);
    assert_eq!(std::fs::read_to_string(outputs.join("findings.md")).unwrap(), "# F-01 critical");
    assert_eq!(doc["guestCopyDiffered"], serde_json::json!(["findings.md", "verdict.json"]));
}
