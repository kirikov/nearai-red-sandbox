//! A controlling PTY for OpenCode, with no path from operator input to the
//! child until the plugin's nonce handshake has succeeded. The launcher pumps
//! this transport itself; no blocking reader threads survive session teardown.
use std::fs::File;
use std::io::{self, Read};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::process::{Child, Command, Stdio};

const BUFFER_LIMIT: usize = 64 * 1024;

pub struct Terminal {
    master: File,
    input: OwnedFd,
    output: OwnedFd,
    enabled: bool,
    input_eof: bool,
    output_eof: bool,
    input_queue: Vec<u8>,
    output_queue: Vec<u8>,
    original: Option<libc::termios>,
}

impl Terminal {
    pub fn spawn(command: &mut Command) -> io::Result<(Child, Self)> {
        Self::spawn_with_io(command, duplicate(0)?, duplicate(1)?)
    }

    fn spawn_with_io(
        command: &mut Command,
        input: OwnedFd,
        output: OwnedFd,
    ) -> io::Result<(Child, Self)> {
        let (master, slave) = pty()?;
        let mut terminal = Self {
            master: File::from(master),
            input,
            output,
            enabled: false,
            input_eof: false,
            output_eof: false,
            input_queue: vec![],
            output_queue: vec![],
            original: None,
        };
        terminal.resize()?;
        // The master is exclusively ours, so changing its file status flags
        // cannot change the operator's stdin/stdout flags.
        let flags = unsafe { libc::fcntl(terminal.master.as_raw_fd(), libc::F_GETFL) };
        if flags < 0
            || unsafe {
                libc::fcntl(
                    terminal.master.as_raw_fd(),
                    libc::F_SETFL,
                    flags | libc::O_NONBLOCK,
                )
            } < 0
        {
            return Err(io::Error::last_os_error());
        }
        command
            .stdin(Stdio::from(duplicate(slave.as_raw_fd())?))
            .stdout(Stdio::from(duplicate(slave.as_raw_fd())?))
            .stderr(Stdio::from(slave));
        // SAFETY: only async-signal-safe libc calls run between fork and exec.
        // std installs the configured stdio descriptors before this hook.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = command.spawn()?;
        Ok((child, terminal))
    }

    /// Queue an OSC 0 window title ahead of any child output. Control
    /// characters are dropped so the title cannot end the escape early.
    pub fn set_title(&mut self, title: &str) {
        let clean: String = title.chars().filter(|c| !c.is_control()).collect();
        self.output_queue.extend_from_slice(format!("\x1b]0;{clean}\x07").as_bytes());
    }

    /// Called only after readiness verification. TCSANOW preserves queued
    /// input; neither terminal input nor terminal responses are read earlier.
    pub fn enable_input(&mut self) -> io::Result<()> {
        if self.enabled {
            return Ok(());
        }
        if let Some(original) = attributes(self.input.as_raw_fd())? {
            let mut raw = original;
            unsafe { libc::cfmakeraw(&mut raw) };
            if unsafe { libc::tcsetattr(self.input.as_raw_fd(), libc::TCSANOW, &raw) } < 0 {
                return Err(io::Error::last_os_error());
            }
            self.original = Some(original);
        }
        self.enabled = true;
        Ok(())
    }

    /// All descriptors are polled before use and the PTY master is nonblocking.
    /// Bounded queues and per-tick I/O keep signal and RSS checks responsive.
    pub fn pump(&mut self) -> io::Result<()> {
        self.resize()?;
        let mut buf = [0; 4096];
        if !self.output_eof && self.output_queue.len() < BUFFER_LIMIT {
            match self.master.read(&mut buf) {
                Ok(0) => self.output_eof = true,
                Ok(n) => self.output_queue.extend_from_slice(&buf[..n]),
                Err(e)
                    if e.kind() == io::ErrorKind::WouldBlock
                        || e.kind() == io::ErrorKind::Interrupted => {}
                // Linux returns EIO when the last slave has closed.
                Err(e) if e.raw_os_error() == Some(libc::EIO) => self.output_eof = true,
                Err(e) => return Err(e),
            }
        }
        flush(self.output.as_raw_fd(), &mut self.output_queue)?;
        if self.enabled
            && !self.input_eof
            && self.input_queue.len() < BUFFER_LIMIT
            && readable(self.input.as_raw_fd())?
        {
            let n =
                unsafe { libc::read(self.input.as_raw_fd(), buf.as_mut_ptr().cast(), buf.len()) };
            if n == 0 {
                self.input_eof = true;
            } else if n > 0 {
                self.input_queue.extend_from_slice(&buf[..n as usize]);
            } else {
                let e = io::Error::last_os_error();
                if e.kind() != io::ErrorKind::Interrupted && e.kind() != io::ErrorKind::WouldBlock {
                    return Err(e);
                }
            }
        }
        if self.enabled && !self.output_eof {
            flush(self.master.as_raw_fd(), &mut self.input_queue)?;
        }
        Ok(())
    }

    fn resize(&mut self) -> io::Result<()> {
        let mut size = libc::winsize {
            ws_row: 24,
            ws_col: 80,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        if unsafe { libc::isatty(self.input.as_raw_fd()) } == 1
            && unsafe { libc::ioctl(self.input.as_raw_fd(), libc::TIOCGWINSZ as _, &mut size) } < 0
        {
            return Err(io::Error::last_os_error());
        }
        if unsafe { libc::ioctl(self.master.as_raw_fd(), libc::TIOCSWINSZ as _, &size) } < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(())
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        if let Some(original) = self.original.take() {
            // Retry a signal-interrupted restoration; the parent still owns
            // this terminal, including when startup/monitoring returns an error.
            while unsafe { libc::tcsetattr(self.input.as_raw_fd(), libc::TCSANOW, &original) } < 0 {
                if io::Error::last_os_error().kind() != io::ErrorKind::Interrupted {
                    break;
                }
            }
        }
    }
}

fn attributes(fd: i32) -> io::Result<Option<libc::termios>> {
    if unsafe { libc::isatty(fd) } != 1 {
        return Ok(None);
    }
    let mut value = std::mem::MaybeUninit::uninit();
    if unsafe { libc::tcgetattr(fd, value.as_mut_ptr()) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(Some(unsafe { value.assume_init() }))
}

fn duplicate(fd: i32) -> io::Result<OwnedFd> {
    let result = unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 3) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { OwnedFd::from_raw_fd(result) })
    }
}

fn pty() -> io::Result<(OwnedFd, OwnedFd)> {
    let (mut master, mut slave) = (-1, -1);
    // SAFETY: both output pointers are valid; optional name/settings are null.
    if unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    } < 0
    {
        return Err(io::Error::last_os_error());
    }
    let pair = unsafe { (OwnedFd::from_raw_fd(master), OwnedFd::from_raw_fd(slave)) };
    for fd in [&pair.0, &pair.1] {
        if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(pair)
}

fn poll(fd: i32, events: i16) -> io::Result<bool> {
    let mut p = libc::pollfd {
        fd,
        events,
        revents: 0,
    };
    let n = unsafe { libc::poll(&mut p, 1, 0) };
    if n < 0 {
        let e = io::Error::last_os_error();
        if e.kind() == io::ErrorKind::Interrupted {
            return Ok(false);
        }
        return Err(e);
    }
    if p.revents & libc::POLLNVAL != 0 {
        // Darwin reports POLLNVAL for regular files and /dev/null even when
        // the descriptor is open; these descriptors need no readiness wait.
        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(fd, stat.as_mut_ptr()) } == 0 {
            let mode = unsafe { stat.assume_init() }.st_mode;
            if mode & libc::S_IFMT == libc::S_IFREG || mode & libc::S_IFMT == libc::S_IFCHR {
                return Ok(true);
            }
        }
        return Err(io::Error::other(
            "terminal descriptor closed or cannot be polled",
        ));
    }
    Ok(n > 0 && p.revents & (events | libc::POLLHUP | libc::POLLERR) != 0)
}

fn readable(fd: i32) -> io::Result<bool> {
    poll(fd, libc::POLLIN)
}

fn flush(fd: i32, queue: &mut Vec<u8>) -> io::Result<()> {
    if queue.is_empty() || !poll(fd, libc::POLLOUT)? {
        return Ok(());
    }
    // Small writes avoid holding the monitor loop behind large terminal output.
    let n = unsafe { libc::write(fd, queue.as_ptr().cast(), queue.len().min(1024)) };
    if n > 0 {
        queue.drain(..n as usize);
    } else if n < 0 {
        let e = io::Error::last_os_error();
        if e.kind() != io::ErrorKind::Interrupted && e.kind() != io::ErrorKind::WouldBlock {
            return Err(e);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::tempdir;
    use std::fs::File;
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    use std::thread;
    use std::time::{Duration, Instant};

    // Removing the input gate lets this real child consume a queued command
    // before the plugin handshake has authenticated its tool hooks.
    #[test]
    fn queued_input_is_inaccessible_until_ready_and_delivered_after_ready() {
        let d = tempdir("terminal-gate");
        let receipt = d.join("receipt");
        // A real operator terminal starts in canonical mode. Queued terminal
        // input must survive the later raw-mode transition without escaping.
        let (host_master, reader) = pty().unwrap();
        let mut writer = File::from(host_master);
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "IFS= read -r line; printf '%s' \"$line\" > \"$1\"",
                "test",
            ])
            .arg(&receipt);
        let output: OwnedFd = File::options()
            .write(true)
            .open("/dev/null")
            .unwrap()
            .into();
        let (mut child, mut terminal) =
            Terminal::spawn_with_io(&mut command, reader, output).unwrap();
        writer.write_all(b"queued command\n").unwrap();
        for _ in 0..20 {
            terminal.pump().unwrap();
            thread::sleep(Duration::from_millis(5));
        }
        let before_ready = std::fs::read(&receipt).ok();
        terminal.enable_input().unwrap();
        let end = Instant::now() + Duration::from_secs(2);
        while child.try_wait().unwrap().is_none() && Instant::now() < end {
            terminal.pump().unwrap();
            thread::sleep(Duration::from_millis(5));
        }
        let after_ready = std::fs::read(&receipt).ok();
        let _ = child.kill();
        child.wait().unwrap();
        std::fs::remove_dir_all(d).unwrap();
        assert_eq!(before_ready, None, "child consumed input before readiness");
        assert_eq!(after_ready, Some(b"queued command".to_vec()));
    }

    #[test]
    fn a_missing_handshake_never_forwards_queued_input() {
        let d = tempdir("terminal-never-ready");
        let receipt = d.join("receipt");
        let (reader, mut writer) = UnixStream::pair().unwrap();
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "IFS= read -r line; printf '%s' \"$line\" > \"$1\"",
                "test",
            ])
            .arg(&receipt);
        let output: OwnedFd = File::options()
            .write(true)
            .open("/dev/null")
            .unwrap()
            .into();
        let (mut child, mut terminal) =
            Terminal::spawn_with_io(&mut command, reader.into(), output).unwrap();
        writer.write_all(b"unsafe host command\n").unwrap();
        let result = crate::session::wait_for_ready(
            &d.join("ready"),
            "expected-nonce",
            Duration::from_millis(100),
            Duration::from_millis(5),
            &mut || {
                terminal.pump().unwrap();
                child.try_wait().unwrap().is_some()
            },
        );
        let receipt_before_teardown = receipt.exists();
        drop(terminal);
        let _ = child.kill();
        child.wait().unwrap();
        let receipt_after_teardown = receipt.exists();
        std::fs::remove_dir_all(d).unwrap();
        assert!(result.unwrap_err().contains("did not report ready"));
        assert!(!receipt_before_teardown && !receipt_after_teardown);
    }

    #[test]
    fn the_launcher_title_reaches_the_operator_terminal_before_child_output() {
        let d = tempdir("terminal-title");
        let out = d.join("out");
        let (reader, _writer) = UnixStream::pair().unwrap();
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "printf child"]);
        let (mut child, mut terminal) =
            Terminal::spawn_with_io(&mut command, reader.into(), File::create(&out).unwrap().into()).unwrap();
        // Control characters cannot end the escape early or inject another.
        terminal.set_title("NEAR AI Red\x07\x1b]0;evil SandBox");
        // Darwin holds an exiting child until its terminal output drains.
        let end = Instant::now() + Duration::from_secs(5);
        while child.try_wait().unwrap().is_none() && Instant::now() < end {
            terminal.pump().unwrap();
            thread::sleep(Duration::from_millis(2));
        }
        for _ in 0..20 {
            terminal.pump().unwrap();
        }
        let _ = child.kill();
        child.wait().unwrap();
        drop(terminal);
        let written = std::fs::read(&out).unwrap();
        std::fs::remove_dir_all(d).unwrap();
        let expected: &[u8] = b"\x1b]0;NEAR AI Red]0;evil SandBox\x07";
        assert!(written.starts_with(expected), "{:?}", String::from_utf8_lossy(&written));
        assert!(String::from_utf8_lossy(&written).contains("child"));
    }

    #[test]
    fn the_child_has_tty_stdio_and_a_controlling_terminal() {
        let d = tempdir("terminal-isatty");
        let receipt = d.join("receipt");
        let (reader, _writer) = UnixStream::pair().unwrap();
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "test -t 0 && test -t 1 && test -t 2 && test -r /dev/tty && printf tty > \"$1\"",
                "test",
            ])
            .arg(&receipt);
        let output: OwnedFd = File::options()
            .write(true)
            .open("/dev/null")
            .unwrap()
            .into();
        let (mut child, _terminal) =
            Terminal::spawn_with_io(&mut command, reader.into(), output).unwrap();
        let status = child.wait().unwrap();
        let result = std::fs::read(&receipt).ok();
        std::fs::remove_dir_all(d).unwrap();
        assert!(status.success());
        assert_eq!(result, Some(b"tty".to_vec()));
    }

    #[test]
    fn raw_mode_starts_only_after_ready_and_is_restored_on_drop() {
        let (_host_master, host_slave) = pty().unwrap();
        let mut initial = attributes(host_slave.as_raw_fd()).unwrap().unwrap();
        initial.c_lflag |= libc::ICANON | libc::ECHO;
        assert_eq!(
            unsafe { libc::tcsetattr(host_slave.as_raw_fd(), libc::TCSANOW, &initial) },
            0
        );
        let before = attributes(host_slave.as_raw_fd()).unwrap().unwrap();
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "IFS= read -r line"]);
        let output: OwnedFd = File::options()
            .write(true)
            .open("/dev/null")
            .unwrap()
            .into();
        let (mut child, mut terminal) = Terminal::spawn_with_io(
            &mut command,
            duplicate(host_slave.as_raw_fd()).unwrap(),
            output,
        )
        .unwrap();
        let before_ready = attributes(host_slave.as_raw_fd()).unwrap().unwrap().c_lflag;
        terminal.enable_input().unwrap();
        let after_ready = attributes(host_slave.as_raw_fd()).unwrap().unwrap().c_lflag;
        drop(terminal);
        let restored = attributes(host_slave.as_raw_fd()).unwrap().unwrap();
        let _ = child.kill();
        child.wait().unwrap();
        assert_eq!(before_ready, before.c_lflag);
        assert_eq!(after_ready & (libc::ICANON | libc::ECHO), 0);
        // Darwin may set PENDIN when canonical mode is restored; it is a
        // kernel queue bookkeeping bit, not an operator mode setting.
        assert_eq!(
            restored.c_lflag & !libc::PENDIN,
            before.c_lflag & !libc::PENDIN
        );
        assert_eq!(restored.c_iflag, before.c_iflag);
        assert_eq!(restored.c_oflag, before.c_oflag);
        assert_eq!(restored.c_cc, before.c_cc);
    }

    #[test]
    fn window_size_is_copied_at_start_and_on_each_pump() {
        let (_host_master, host_slave) = pty().unwrap();
        let set = |rows, cols| {
            let size = libc::winsize {
                ws_row: rows,
                ws_col: cols,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            assert_eq!(
                unsafe { libc::ioctl(host_slave.as_raw_fd(), libc::TIOCSWINSZ as _, &size) },
                0
            );
        };
        set(37, 101);
        let mut command = Command::new("/bin/sh");
        command.args(["-c", "IFS= read -r line"]);
        let output: OwnedFd = File::options()
            .write(true)
            .open("/dev/null")
            .unwrap()
            .into();
        let (mut child, mut terminal) = Terminal::spawn_with_io(
            &mut command,
            duplicate(host_slave.as_raw_fd()).unwrap(),
            output,
        )
        .unwrap();
        let size = |terminal: &Terminal| {
            let mut value = libc::winsize {
                ws_row: 0,
                ws_col: 0,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            assert_eq!(
                unsafe {
                    libc::ioctl(
                        terminal.master.as_raw_fd(),
                        libc::TIOCGWINSZ as _,
                        &mut value,
                    )
                },
                0
            );
            (value.ws_row, value.ws_col)
        };
        let initial = size(&terminal);
        set(43, 119);
        terminal.pump().unwrap();
        let changed = size(&terminal);
        drop(terminal);
        let _ = child.kill();
        child.wait().unwrap();
        assert_eq!(initial, (37, 101));
        assert_eq!(changed, (43, 119));
    }

    /// No VM or credentials: validates the pinned application's interactive
    /// startup through the real sandbox plugin, then exits without a prompt.
    #[test]
    #[ignore = "requires RTS_OPENCODE pointing to opencode 2.2.2"]
    fn pinned_opencode_tui_starts_and_handshakes_over_the_gated_pty() {
        use crate::session::{
            TERMINAL_TITLE, opencode_launch, prepare_session_dirs, session_paths, wait_for_ready,
            write_opencode_config, write_tui_config,
        };
        use serde_json::json;
        const FOOTER: &str = "NEAR AI Red · SandBox · test-no-inference · attested";
        let binary =
            std::env::var("RTS_OPENCODE").expect("set RTS_OPENCODE to the pinned 2.2.2 binary");
        let d = tempdir("terminal-opencode");
        let paths = session_paths("0123456789ab", &d.join("sessions"), &d.join("runtime"));
        prepare_session_dirs(&paths).unwrap();
        let nonce = "0123456789abcdef0123456789abcdef";
        let provider = crate::config::ProviderConfig {
            id: "ironwire".into(),
            base_url: "http://127.0.0.1:1/v1".into(),
            models: vec!["test-no-inference".into()],
            api_key_env: None,
            extra_body: None,
        };
        let plugin_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("plugin");
        write_tui_config(
            &paths.config_dir,
            &plugin_dir,
            &crate::config::model_display_name(&provider.models[0]),
            true,
        )
        .unwrap();
        crate::brand::theme::install(&plugin_dir, &paths.config_dir).unwrap();
        write_opencode_config(&paths.config_dir, &plugin_dir, &json!({
            "machine": paths.machine, "nonce": nonce, "readyFile": paths.ready_file,
            "auditFile": paths.audit_file, "provider": "ironwire", "models": provider.models, "rts": "/nonexistent-rts-no-tools-called",
            "bridge": { "machine": paths.machine }, "debugFile": paths.runtime.join("debug.json"),
        }), &provider, false, None).unwrap();
        let launch = opencode_launch(
            &paths,
            &binary,
            [
                ("PATH".into(), "/usr/bin:/bin:/usr/sbin:/sbin".into()),
                ("TERM".into(), "xterm-256color".into()),
                ("COLORTERM".into(), "truecolor".into()),
            ],
        );
        let version = Command::new(&binary)
            .arg("--version")
            .env_clear()
            .envs(launch.env.iter().cloned())
            .output()
            .unwrap();
        assert!(
            version.status.success(),
            "{}",
            String::from_utf8_lossy(&version.stderr)
        );
        assert!(
            String::from_utf8_lossy(&version.stdout)
                .split_whitespace()
                .any(|part| part.trim_start_matches('v') == "2.2.2")
        );
        let (reader, mut writer) = UnixStream::pair().unwrap();
        let output_file = File::create(paths.runtime.join("terminal-output")).unwrap();
        let mut command = Command::new(&binary);
        command
            .args(&launch.argv[1..])
            .current_dir(&launch.cwd)
            .env_clear()
            .envs(launch.env.iter().cloned());
        let (mut child, mut terminal) =
            Terminal::spawn_with_io(&mut command, reader.into(), output_file.into()).unwrap();
        terminal.set_title(TERMINAL_TITLE);
        let ready = wait_for_ready(
            &paths.ready_file,
            nonce,
            Duration::from_secs(25),
            Duration::from_millis(10),
            &mut || {
                terminal.pump().unwrap();
                child.try_wait().unwrap().is_some()
            },
        );
        let mut exited_from_input = false;
        if ready.is_ok() {
            terminal.enable_input().unwrap();
            // Plugin initialization finishes before the TUI starts reading.
            // Wait for the home screen's footer (at most 15 s), then send
            // spaced Ctrl-C keys; no prompt is submitted.
            let mut home_drawn_at = None;
            for tick in 0..2000 {
                terminal.pump().unwrap();
                if home_drawn_at.is_none() && tick % 20 == 0 {
                    let so_far = std::fs::read(paths.runtime.join("terminal-output")).unwrap();
                    if tick >= 1500 || screen_text_and_titles(&so_far).0.contains(FOOTER) {
                        home_drawn_at = Some(tick);
                    }
                }
                if home_drawn_at.is_some_and(|at| tick >= at + 20 && tick % 50 == 0) {
                    writer.write_all(b"\x03").unwrap();
                }
                if child.try_wait().unwrap().is_some() {
                    exited_from_input = true;
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
        }
        crate::sys::send_signal(-(child.id() as i32), crate::sys::SIGKILL);
        let _ = child.wait();
        drop(terminal);
        let output = std::fs::read(paths.runtime.join("terminal-output")).unwrap();
        let debug = std::fs::read_to_string(paths.runtime.join("debug.json")).ok();
        if let Err(e) = ready {
            panic!("{e}; PTY output: {}", String::from_utf8_lossy(&output));
        }
        assert!(
            exited_from_input,
            "OpenCode did not respond to after-ready terminal input"
        );
        assert!(
            debug.is_some(),
            "real plugin did not verify the tool, MCP and model state"
        );
        assert!(
            !output.is_empty(),
            "OpenCode emitted no interactive terminal output"
        );
        let (text, titles) = screen_text_and_titles(&output);
        // The branding TUI plugin drew the home footer with its options.
        assert!(text.contains(FOOTER), "footer missing; screen text: {text}");
        // rts set the title first; with terminal.title off opencode sets none
        // (at most the empty title its renderer writes on exit).
        assert_eq!(titles.first().map(String::as_str), Some(TERMINAL_TITLE), "{titles:?}");
        assert!(
            titles.iter().all(|t| t == TERMINAL_TITLE || t.is_empty()),
            "{titles:?}"
        );
        // The nearai-red theme is active: opencode's own muted hints use its
        // pewter textMuted and body text its TOP (opencode's default theme
        // has neither), on the terminal's own background.
        let raw = String::from_utf8_lossy(&output);
        assert!(raw.contains("\x1b[38;2;128;128;136m\x1b[49magents"), "theme textMuted missing");
        assert!(raw.contains("\x1b[38;2;233;231;225m\x1b[49mctrl+p"), "theme text missing");
        // The footer is drawn in sand, with "attested" in NEAR AI Red.
        assert!(
            raw.contains("\x1b[38;2;161;151;127m\x1b[49mNEAR AI Red · SandBox · test-no-inference · "),
            "footer not in sand"
        );
        assert!(raw.contains("\x1b[38;2;235;25;36m\x1b[49mattested"), "attested not in red");
        eprintln!("footer found; title escapes: {titles:?}");
        if let Ok(dump) = std::env::var("RTS_PTY_DUMP") {
            std::fs::write(dump, &output).unwrap();
        }
        std::fs::remove_dir_all(d).unwrap();
    }

    /// Terminal output with CSI and OSC escapes removed, and the payloads of
    /// the OSC 0 and OSC 2 (window title) escapes in order.
    fn screen_text_and_titles(output: &[u8]) -> (String, Vec<String>) {
        let (mut text, mut titles) = (Vec::new(), Vec::new());
        let mut i = 0;
        while i < output.len() {
            match (output[i], output.get(i + 1)) {
                (0x1b, Some(b'[')) => {
                    i += 2;
                    while i < output.len() && !(0x40..=0x7e).contains(&output[i]) {
                        i += 1;
                    }
                    i += 1;
                }
                (0x1b, Some(b']')) => {
                    let start = i + 2;
                    let mut end = start;
                    while end < output.len()
                        && output[end] != 0x07
                        && !(output[end] == 0x1b && output.get(end + 1) == Some(&b'\\'))
                    {
                        end += 1;
                    }
                    let body = String::from_utf8_lossy(&output[start..end]);
                    if let Some(title) = body.strip_prefix("0;").or_else(|| body.strip_prefix("2;")) {
                        titles.push(title.to_string());
                    }
                    i = end + if output.get(end) == Some(&0x07) { 1 } else { 2 };
                }
                (b, _) => {
                    text.push(b);
                    i += 1;
                }
            }
        }
        (String::from_utf8_lossy(&text).into_owned(), titles)
    }
}
