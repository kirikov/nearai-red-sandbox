# Container backend for Linux hosts: design

Date: 2026-11-02
Status: proposed; implemented as `src/docker.rs` behind the existing `Sandbox` seam, with the headless session in `src/headless.rs` (`rts run`).

## Purpose

Run the same sandboxed opencode sessions on a Linux host that has no
hypervisor of its own: a confidential VM (Intel TDX under dstack, as on
Phala Cloud or NEAR AI Cloud), where nested virtualization is unavailable
and mvm cannot start a microVM. The wall between a session and the host
becomes the container runtime, with the TEE around the whole VM shielding
code and credentials from the cloud operator and attesting the image that
ran. This is the runner for the hosted NEAR AI Red review agent; the Mac
product is unchanged.

## What was measured (dstack 0.6, tdx.small, 2026-11-02)

- The Docker daemon is reachable through its socket from a compose service;
  sibling containers with `--network none`, read-only rootfs and no
  capabilities start and behave; cgroup limits and tmpfs hold.
- Runtimes: `runc` and `sysbox-runc`. No `runsc`, no `/dev/kvm`.
- Images build inside the VM (classic and BuildKit).
- Egress from the VM is open by default; in-guest nftables is the
  supported way to restrict it.

## Decisions

| Topic | Decision |
|---|---|
| Seam | `DockerSandbox` implements `Exec` and `Sandbox`; the wrapper protocol, the bridge, copyIn/copyOut and the hostile-tar extractor are reused unchanged. |
| Runtime | `--runtime sysbox-runc` where the daemon offers it (user namespace, virtualized procfs and sysfs), the daemon default otherwise. Chosen per session, validated as a plain token. |
| Hardening | `--network none --read-only --cap-drop ALL --security-opt no-new-privileges --pids-limit 1024 --memory/--memory-swap --cpus --tmpfs /tmp --init`. PID 1 is a `sh` keepalive. |
| Image | A repo digest (`name@sha256:…`) or a local image id (`sha256:…`). Tags refused, as for mvm. The guest image is `guest/Dockerfile.linux-amd64`: the same layout as the microVM guest (`/in` and `/out` link into `/tmp/rts`, which `initialize_guest_storage` insists on), on the amd64 Alpine digest; `guest/Dockerfile` pins the arm64 Alpine for the Apple-silicon microVM and does not run on x86 hosts. |
| Network sessions | Refused by `create`. The container has no egress; an allowlist for the controller that drives sessions is the controller's job, outside this backend. |
| Exec semantics | No serialization: Docker does not journal execs and a nonzero guest exit does not poison the container. A missing trailer, a host-backstop kill, or a daemon error naming a gone container marks the sandbox unavailable for good, as mvm does. |
| Bridge | `rts tool` takes `bridge.backend: "mvm" \| "docker"` (default `mvm`), `docker` (CLI path) and `runtime`. `mvmctl` and `emptyGitConfig` are required only for mvm. The plugin passes the object through unchanged. |
| Tests | Unit tests with a fake runner mirror `mvm.rs`. `tests/container.rs` runs the containment checks against a real daemon, ignored by default, prerequisites fail loudly. |
| State dir | The container sandbox shares the session's state directory like mvm: any process's unavailable marker refuses later execs, and execs hold the exec lock shared (no serialization among themselves) so teardown's exclusive `quiesce` still waits for them. |
| copyIn | The guest tar extracts with `-o`: a container without `CAP_CHOWN` cannot restore the archive's owner, and both busybox and GNU tar accept the flag. |
| Headless session | `rts run --image REF --prompt-file FILE [--inputs DIR] [--docker PATH] [--runtime NAME] [--timeout-min N] --allow-unattested-provider`: the session file supplies provider, limits, opencode and the plugin; the sandbox is a container with no network; `opencode run --standalone --format json --auto` runs the prompt once; its JSON events land in `sessions/<id>/opencode.jsonl`; `/out` is exported at teardown and `session.json` records the outcome. An explicit provider is required: the attested embedded provider (IronWire and the pinned forwarder) is not wired for headless yet, so the run is recorded unattested like `rts launch --allow-unattested-provider`. |

## Not in this increment

- Attested inference for the headless session (the forwarder and IronWire
  start on the `launch` path only).
- An RSS ceiling for containers in the session loop; the orphan sweep only
  logs `rts-*` containers it finds, as the mvm sweep does.
- Building the guest image for Linux in CI and publishing its digest.
