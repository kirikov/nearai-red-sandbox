# Container backend for Linux hosts: design

Date: 2026-11-02
Status: proposed; first increment implemented as `src/docker.rs` behind the existing `Sandbox` seam.

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

## Not in this increment

- A headless session (`rts run`): the launcher still assumes the TUI and the
  mvm preflight. It is the next change on this branch.
- Orphan sweep and RSS ceiling for containers in the session loop.
- Building the guest image for Linux in CI and publishing its digest.
