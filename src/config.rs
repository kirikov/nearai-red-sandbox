//! Operator config for one session, read from a JSON file.
//!
//! By default the launcher signs in to NEAR AI Cloud and starts an embedded
//! IronWire proxy. Explicit provider connection fields select an external
//! endpoint instead. OpenCode receives only a placeholder key in embedded mode.

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::mvm::is_image_ref;

pub const DEFAULT_IMAGE: &str = "us-west1-docker.pkg.dev/nearai-red-sandbox/sandbox-images/guest@sha256:56406743ca98389029c795a9cd55f0d1f55e569d57226c10e6c4f6f340cf9ac1";
pub const IRONWIRE_PROVIDER_ID: &str = "ironwire";
/// The provider rts itself configures after attestation (src/inference.rs).
pub const EMBEDDED_PROVIDER_ID: &str = "nearai-red-embedded";
/// The provider rts configures on the headless direct path: opencode talks to
/// the forwarder itself, which holds the key (rts run --inference direct).
pub const DIRECT_PROVIDER_ID: &str = "nearai-red-direct";

/// How a session's inference reached NEAR AI, by its provider: "ironwire"
/// (IronWire holds the key, in front of the forwarder), "direct" (the
/// forwarder holds the key), or "external" (an operator-named provider).
pub fn inference_mode(provider_id: &str) -> &'static str {
    match provider_id {
        EMBEDDED_PROVIDER_ID => "ironwire",
        DIRECT_PROVIDER_ID => "direct",
        _ => "external",
    }
}
pub const IRONWIRE_BASE_URL: &str = "http://127.0.0.1:8463/openai/v1";
/// Sent as the API key when the provider needs none. IronWire, or the
/// forwarder in direct mode, strips the inbound authorization header and
/// attaches its own credential.
pub const PLACEHOLDER_API_KEY: &str = "ironwire-holds-the-key";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderConfig {
    pub id: String,
    pub base_url: String,
    pub models: Vec<String>,
    /// Environment variable holding the key, for a provider reached
    /// directly. None for IronWire.
    pub api_key_env: Option<String>,
}

impl ProviderConfig {
    /// Host and port when the provider is on this machine's loopback.
    pub fn loopback(&self) -> Option<(String, u16, String)> {
        let rest = self.base_url.strip_prefix("http://")?;
        let (authority, path) = rest
            .split_once('/')
            .map_or((rest, String::new()), |(a, p)| (a, format!("/{p}")));
        let (host, port) = authority.rsplit_once(':')?;
        if host != "127.0.0.1" && host != "localhost" {
            return None;
        }
        Some((host.to_string(), port.parse().ok()?, path))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub tool_timeout_ms: u64,
    pub max_output_bytes: u64,
    pub memory_mib: u32,
    pub cpus: u32,
    pub host_rss_ceiling_mib: u64,
    pub handshake_ms: u64,
    /// Wall-clock limit from boot, in hours; also the attach `--ttl`.
    pub session_hours: u32,
    /// opencode's context window for the model, in tokens; drives auto-compaction.
    /// None leaves it to opencode's provider default.
    pub context_tokens: Option<u64>,
    /// The session file set `hostRssCeilingMiB`; otherwise it follows memory.
    rss_ceiling_set: bool,
}

/// mvm v0.22.0's inclusive `MAX_TTL` (30 days), in hours.
pub const MAX_SESSION_HOURS: u32 = 720;

/// The default host RSS ceiling is the guest's memory plus this much, for
/// the HVF supervisor's own overhead.
pub const RSS_HEADROOM_MIB: u64 = 1024;

/// mvm's floor for guest memory.
pub const MIN_MEMORY_MIB: u32 = 64;

impl Default for Limits {
    fn default() -> Self {
        Self {
            tool_timeout_ms: 120_000,
            max_output_bytes: 65_536,
            memory_mib: 2048,
            cpus: 2,
            host_rss_ceiling_mib: 2048 + RSS_HEADROOM_MIB,
            handshake_ms: 30_000,
            session_hours: 24,
            context_tokens: None,
            rss_ceiling_set: false,
        }
    }
}

impl Limits {
    /// Apply `--cpus` and `--memory`. A ceiling the session file did not set
    /// moves with memory; one it did set must still cover memory.
    pub fn resize(&mut self, cpus: Option<u32>, memory_mib: Option<u32>) -> Result<(), String> {
        if let Some(c) = cpus {
            self.cpus = c;
        }
        if let Some(m) = memory_mib {
            self.memory_mib = m;
            if !self.rss_ceiling_set {
                self.host_rss_ceiling_mib = u64::from(m) + RSS_HEADROOM_MIB;
            } else if u64::from(m) > self.host_rss_ceiling_mib {
                return Err(format!(
                    "--memory {m} MiB is above limits.hostRssCeilingMiB ({}) in the session file; raise or remove that limit",
                    self.host_rss_ceiling_mib
                ));
            }
        }
        Ok(())
    }
}

/// `--memory`: whole MiB, as `N`, `NM` or `NG` (either case), at least 64 MiB.
pub fn parse_memory(s: &str) -> Result<u32, String> {
    let bad = || format!("--memory must be a whole number of MiB, or end in M or G (for example 8192 or 8G), and at least {MIN_MEMORY_MIB} MiB: {s}");
    let (digits, scale) = match s.as_bytes().last() {
        Some(b'G' | b'g') => (&s[..s.len() - 1], 1024),
        Some(b'M' | b'm') => (&s[..s.len() - 1], 1),
        _ => (s, 1),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(bad());
    }
    let mib = digits
        .parse::<u32>()
        .ok()
        .and_then(|n| n.checked_mul(scale))
        .ok_or_else(bad)?;
    if mib < MIN_MEMORY_MIB {
        return Err(bad());
    }
    Ok(mib)
}

/// Refuse a VM larger than the host: more vCPUs than logical CPUs, or more
/// memory than physical RAM.
pub fn check_host_fit(l: &Limits, host_cpus: u32, host_memory_mib: u64) -> Result<(), String> {
    if l.cpus > host_cpus {
        return Err(format!("the VM asks for {} vCPUs but this host has {host_cpus}", l.cpus));
    }
    if u64::from(l.memory_mib) > host_memory_mib {
        return Err(format!(
            "the VM asks for {} MiB but this host has {host_memory_mib} MiB",
            l.memory_mib
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperatorConfig {
    pub embedded_inference: bool,
    pub image: String,
    pub inputs_dir: Option<PathBuf>,
    pub provider: ProviderConfig,
    pub limits: Limits,
    /// Where sessions/<id>/out and tools.jsonl go.
    pub sessions_dir: PathBuf,
    pub opencode: String,
    pub mvmctl: PathBuf,
    /// The opencode plugin shim (a directory holding package.json).
    pub plugin_dir: PathBuf,
    /// Guest egress allowlist entries as written (`network.allow`, then any
    /// `--allow-host` flags). Empty means no network.
    pub network: Vec<String>,
    /// `"updateCheck"`: the launch update check (default on).
    pub update_check: bool,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawProvider {
    id: Option<String>,
    #[serde(rename = "baseURL")]
    base_url: Option<String>,
    models: Option<Vec<String>>,
    api_key_env: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct RawLimits {
    tool_timeout_ms: Option<u64>,
    max_output_bytes: Option<u64>,
    #[serde(rename = "memoryMiB")]
    memory_mib: Option<u32>,
    cpus: Option<u32>,
    #[serde(rename = "hostRssCeilingMiB")]
    host_rss_ceiling_mib: Option<u64>,
    handshake_ms: Option<u64>,
    #[serde(rename = "contextTokens")]
    context_tokens: Option<u64>,
    /// Read as any JSON value so a string or fraction gets this module's
    /// refusal, not serde's.
    session_hours: Option<serde_json::Value>,
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct RawNetwork {
    allow: Option<Vec<String>>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Raw {
    image: Option<String>,
    inputs_dir: Option<String>,
    #[serde(default)]
    provider: RawProvider,
    #[serde(default)]
    limits: RawLimits,
    sessions_dir: Option<String>,
    opencode: Option<String>,
    mvmctl: Option<String>,
    plugin_dir: Option<String>,
    #[serde(default)]
    network: RawNetwork,
    update_check: Option<bool>,
}

fn is_env_name(s: &str) -> bool {
    let mut b = s.bytes();
    matches!(b.next(), Some(c) if c.is_ascii_uppercase() || c == b'_')
        && b.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
}

fn reserved_credential_env(name: &str) -> bool {
    // A direct provider may preserve exactly this variable when constructing
    // OpenCode's environment. It must never be an execution/configuration
    // control that could undo the session's isolated home or runtime policy.
    [
        "XDG_",
        "OPENCODE_",
        "RTS_",
        "LC_",
        "GIT_",
        "NODE_",
        "BUN_",
        "LD_",
        "DYLD_",
        "NPM_CONFIG_",
        "PYTHON",
        "RUBY",
        "PERL",
    ]
    .iter()
    .any(|prefix| name.starts_with(prefix))
        || matches!(
            name,
            "HOME"
                | "PATH"
                | "PWD"
                | "OLDPWD"
                | "USER"
                | "LOGNAME"
                | "SHELL"
                | "ENV"
                | "BASH_ENV"
                | "ZDOTDIR"
                | "FPATH"
                | "CDPATH"
                | "IFS"
                | "PS4"
                | "PROMPT_COMMAND"
                | "TMPDIR"
                | "TMP"
                | "TEMP"
                | "LANG"
                | "LANGUAGE"
                | "TZ"
                | "TERM"
                | "TERMINFO"
                | "TERMINFO_DIRS"
                | "HTTP_PROXY"
                | "HTTPS_PROXY"
                | "ALL_PROXY"
                | "NO_PROXY"
                | "SSL_CERT_FILE"
                | "SSL_CERT_DIR"
                | "CURL_CA_BUNDLE"
                | "REQUESTS_CA_BUNDLE"
                | "_"
        )
}

fn positive<T: Copy + Default + PartialOrd>(v: Option<T>, d: T, k: &str) -> Result<T, String> {
    let v = v.unwrap_or(d);
    if v <= T::default() {
        Err(format!("config: limits.{k} must be a positive integer"))
    } else {
        Ok(v)
    }
}

/// `limits.sessionHours`: absent is the default, otherwise an integer from
/// 1 to [`MAX_SESSION_HOURS`].
fn session_hours(v: Option<&serde_json::Value>, d: u32) -> Result<u32, String> {
    let Some(v) = v else { return Ok(d) };
    v.as_u64()
        .filter(|h| (1..=u64::from(MAX_SESSION_HOURS)).contains(h))
        .and_then(|h| u32::try_from(h).ok())
        .ok_or_else(|| format!("config: limits.sessionHours must be an integer from 1 to {MAX_SESSION_HOURS}"))
}

/// Paths the config leaves out, from `install::defaults`.
pub struct Defaults {
    pub mvmctl: PathBuf,
    pub plugin_dir: PathBuf,
    pub opencode: String,
    pub sessions_dir: PathBuf,
}

impl Defaults {
    /// From HOME (empty if unset), RTS_MVMCTL, the install prefix of the
    /// running binary, and the checkout it was built from.
    pub fn from_env() -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default();
        let rts_mvmctl = std::env::var_os("RTS_MVMCTL").map(PathBuf::from);
        let prefix = crate::install::current_prefix();
        crate::install::defaults(&crate::install::Lookup {
            source: Path::new(env!("CARGO_MANIFEST_DIR")),
            prefix: prefix.as_deref(),
            home: &home,
            rts_mvmctl: rts_mvmctl.as_deref(),
        })
    }
}

/// The route ID that stood in for the default model before NEAR AI's
/// production launch. It is no longer a default, but a config written then
/// may still name it, so launching with it is still refused.
pub const MODEL_ID_PLACEHOLDER: &str = "PLACEHOLDER-near-private-abliterated";
/// NEAR AI's abliterated DeepSeek, listed by production `/v1/models` on
/// 2026-10-29 (owned_by `nearai`).
pub const DEFAULT_MODELS: &[&str] = &["deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated"];

const MAX_DISPLAY_CHARS: usize = 64;

/// What the operator sees for a model in opencode. Display only: the
/// allowlist, plugin checks, routing headers and forwarder use the ID.
/// An operator-config ID is only checked non-empty, so the name is cleaned:
/// control and bidi-format characters could reach the terminal or reorder
/// text, and a footer separator could fake the attestation segment.
pub fn model_display_name(id: &str) -> String {
    let raw = match id {
        "deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated" => "DeepSeek V4.1 Flash Abliterated",
        _ => match id.rsplit_once('/') {
            Some((_, last)) if !last.is_empty() => last,
            _ => id,
        },
    };
    let spaced: String = raw.chars().map(|c| if display_unsafe(c) { ' ' } else { c }).collect();
    let clean: String = spaced.split_whitespace().collect::<Vec<_>>().join(" ").chars().take(MAX_DISPLAY_CHARS).collect();
    match clean.trim_end() {
        "" => "unnamed model".into(),
        name => name.into(),
    }
}

fn display_unsafe(c: char) -> bool {
    c.is_control()
        || matches!(c, '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        // The footer separator and its look-alikes.
        || matches!(c, '·' | '•' | '‧' | '∙' | '⋅' | '・' | '･')
}

pub fn require_real_models(models: &[String]) -> Result<(), String> {
    if models.iter().any(|m| m == MODEL_ID_PLACEHOLDER) {
        return Err(format!(
            "provider.models still contains the retired placeholder {MODEL_ID_PLACEHOLDER}: remove it from provider.models, or omit provider.models for the default NEAR AI model"
        ));
    }
    Ok(())
}

pub fn parse_config(text: &str, base_dir: &Path, d: &Defaults) -> Result<OperatorConfig, String> {
    let raw: Raw = serde_json::from_str(text).map_err(|e| format!("config: {e}"))?;
    let p = raw.provider;
    let embedded_inference = p.id.is_none() && p.base_url.is_none() && p.api_key_env.is_none();
    if let Some(id @ (EMBEDDED_PROVIDER_ID | DIRECT_PROVIDER_ID)) = p.id.as_deref() {
        return Err(format!(
            "config: provider.id {id} is reserved for rts's attested NEAR AI inference; omit provider for it, or name an external provider differently"
        ));
    }
    let id = p.id.unwrap_or_else(|| IRONWIRE_PROVIDER_ID.into());
    if id.is_empty()
        || !id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
    {
        return Err("config: provider.id must be lowercase letters, digits and dashes".into());
    }
    let base_url = p.base_url.unwrap_or_else(|| IRONWIRE_BASE_URL.into());
    if let Some(k) = &p.api_key_env
        && !is_env_name(k)
    {
        return Err("config: provider.apiKeyEnv must be an environment variable name".into());
    }
    if p.api_key_env
        .as_deref()
        .is_some_and(reserved_credential_env)
    {
        return Err(
            "config: provider.apiKeyEnv cannot name a reserved session or runtime variable".into(),
        );
    }
    let models = p
        .models
        .unwrap_or_else(|| DEFAULT_MODELS.iter().map(|m| (*m).to_string()).collect());
    if models.is_empty() || models.iter().any(String::is_empty) {
        return Err("config: provider.models must be a non-empty list of model IDs".into());
    }
    let provider = ProviderConfig {
        id,
        base_url,
        models,
        api_key_env: p.api_key_env,
    };
    // Plain http only to loopback: anything else would send prompts in the clear.
    if !provider.base_url.starts_with("https://") && provider.loopback().is_none() {
        return Err("config: provider.baseURL must be https, or http on 127.0.0.1 or localhost with a port (IronWire)".into());
    }
    if provider.api_key_env.is_none() && provider.loopback().is_none() {
        return Err(
            "config: provider.apiKeyEnv is required for a provider that is not on loopback".into(),
        );
    }
    let image = raw.image.unwrap_or_else(|| DEFAULT_IMAGE.into());
    if !is_image_ref(&image) {
        return Err(
            "config: image must be a digest-pinned reference (name@sha256:<64 hex>)".into(),
        );
    }
    let dl = Limits::default();
    let l = raw.limits;
    let memory_mib = positive(l.memory_mib, dl.memory_mib, "memoryMiB")?;
    let host_rss_ceiling_mib = positive(
        l.host_rss_ceiling_mib,
        u64::from(memory_mib) + RSS_HEADROOM_MIB,
        "hostRssCeilingMiB",
    )?;
    if host_rss_ceiling_mib < u64::from(memory_mib) {
        return Err(format!(
            "config: limits.hostRssCeilingMiB ({host_rss_ceiling_mib}) must be at least limits.memoryMiB ({memory_mib}); omit it to use memoryMiB + {RSS_HEADROOM_MIB}"
        ));
    }
    let limits = Limits {
        tool_timeout_ms: positive(l.tool_timeout_ms, dl.tool_timeout_ms, "toolTimeoutMs")?,
        max_output_bytes: positive(l.max_output_bytes, dl.max_output_bytes, "maxOutputBytes")?,
        memory_mib,
        cpus: positive(l.cpus, dl.cpus, "cpus")?,
        host_rss_ceiling_mib,
        handshake_ms: positive(l.handshake_ms, dl.handshake_ms, "handshakeMs")?,
        session_hours: session_hours(l.session_hours.as_ref(), dl.session_hours)?,
        context_tokens: match l.context_tokens {
            Some(n) if n == 0 => return Err("config: limits.contextTokens must be a positive integer".into()),
            other => other,
        },
        rss_ceiling_set: l.host_rss_ceiling_mib.is_some(),
    };
    let network = raw.network.allow.unwrap_or_default();
    crate::network::expand(&network).map_err(|e| format!("config: network.allow: {e}"))?;
    let path = |v: Option<String>, k: &str| -> Result<Option<PathBuf>, String> {
        match v {
            None => Ok(None),
            Some(s) if s.is_empty() => Err(format!("config: {k} must be a path")),
            Some(s) => Ok(Some(base_dir.join(s))),
        }
    };
    Ok(OperatorConfig {
        embedded_inference,
        image,
        inputs_dir: path(raw.inputs_dir, "inputsDir")?,
        provider,
        limits,
        sessions_dir: path(raw.sessions_dir, "sessionsDir")?
            .unwrap_or_else(|| d.sessions_dir.clone()),
        opencode: raw
            .opencode
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| d.opencode.clone()),
        mvmctl: path(raw.mvmctl, "mvmctl")?.unwrap_or_else(|| d.mvmctl.clone()),
        plugin_dir: path(raw.plugin_dir, "pluginDir")?.unwrap_or_else(|| d.plugin_dir.clone()),
        network,
        update_check: raw.update_check.unwrap_or(true),
    })
}

pub fn load_config(file: &Path, d: &Defaults) -> Result<OperatorConfig, String> {
    let text = std::fs::read_to_string(file)
        .map_err(|e| format!("config: cannot read {}: {e}", file.display()))?;
    let abs = std::path::absolute(file).map_err(|e| format!("config: {e}"))?;
    parse_config(&text, abs.parent().unwrap_or(Path::new("/")), d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d() -> Defaults {
        Defaults {
            mvmctl: "/home/op/.local/mvm/v0.22.0/mvmctl".into(),
            plugin_dir: "/repo/plugin".into(),
            opencode: "opencode".into(),
            sessions_dir: "/repo/sessions".into(),
        }
    }

    #[test]
    fn update_check_defaults_on_and_can_be_turned_off() {
        assert!(parse("{}").unwrap().update_check);
        assert!(!parse(r#"{ "updateCheck": false }"#).unwrap().update_check);
        let e = parse(r#"{ "updateCheck": "no" }"#).unwrap_err();
        assert!(e.starts_with("config:"), "{e}");
    }

    #[test]
    fn an_omitted_provider_uses_the_default_near_ai_model() {
        let c = parse("{}").unwrap();
        assert!(c.embedded_inference);
        assert_eq!(c.provider.models, DEFAULT_MODELS);
        assert_eq!(c.inputs_dir, None);
        assert_eq!(c.provider.api_key_env, None);
    }
    fn parse(t: &str) -> Result<OperatorConfig, String> {
        parse_config(t, Path::new("/cfg"), &d())
    }

    #[test]
    fn the_rss_ceiling_follows_memory_unless_set() {
        let p = |l: &str| parse(&format!(r#"{{ "limits": {l} }}"#));
        let l = p("{}").unwrap().limits;
        assert_eq!((l.memory_mib, l.host_rss_ceiling_mib), (2048, 3072));
        assert_eq!(Limits::default().host_rss_ceiling_mib, 3072);
        let l = p(r#"{ "memoryMiB": 8192 }"#).unwrap().limits;
        assert_eq!(l.host_rss_ceiling_mib, 8192 + RSS_HEADROOM_MIB);
        let l = p(r#"{ "memoryMiB": 8192, "hostRssCeilingMiB": 10000 }"#).unwrap().limits;
        assert_eq!(l.host_rss_ceiling_mib, 10000);
        assert_eq!(
            p(r#"{ "memoryMiB": 8192, "hostRssCeilingMiB": 4096 }"#).unwrap_err(),
            "config: limits.hostRssCeilingMiB (4096) must be at least limits.memoryMiB (8192); omit it to use memoryMiB + 1024"
        );
    }

    #[test]
    fn resize_overrides_size_and_moves_only_a_default_ceiling() {
        let mut l = Limits::default();
        l.resize(Some(4), Some(8192)).unwrap();
        assert_eq!((l.cpus, l.memory_mib, l.host_rss_ceiling_mib), (4, 8192, 9216));
        l.resize(None, None).unwrap();
        assert_eq!((l.cpus, l.memory_mib, l.host_rss_ceiling_mib), (4, 8192, 9216));
        let p = |t: &str| parse(t).unwrap().limits;
        let mut l = p(r#"{ "limits": { "hostRssCeilingMiB": 6000 } }"#);
        l.resize(None, Some(4096)).unwrap();
        assert_eq!(l.host_rss_ceiling_mib, 6000);
        let e = l.resize(None, Some(8192)).unwrap_err();
        assert!(e.contains("--memory 8192 MiB is above limits.hostRssCeilingMiB (6000)"), "{e}");
    }

    #[test]
    fn memory_sizes_parse_as_mib() {
        for (s, mib) in [("64", 64), ("8192", 8192), ("8192M", 8192), ("8192m", 8192), ("8G", 8192), ("1g", 1024)] {
            assert_eq!(parse_memory(s), Ok(mib), "{s}");
        }
        for bad in ["", "0", "63", "8GB", "8 G", "1.5G", "-1", "G", "8T", "4194304G", "99999999999"] {
            assert!(parse_memory(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn host_fit_refuses_more_than_the_host_has() {
        let mut l = Limits::default();
        assert_eq!(check_host_fit(&l, 8, 16384), Ok(()));
        l.resize(Some(9), None).unwrap();
        assert_eq!(check_host_fit(&l, 8, 16384).unwrap_err(), "the VM asks for 9 vCPUs but this host has 8");
        l.resize(Some(8), Some(16384)).unwrap();
        assert_eq!(check_host_fit(&l, 8, 16384), Ok(()));
        l.resize(None, Some(16385)).unwrap();
        assert_eq!(check_host_fit(&l, 8, 16384).unwrap_err(), "the VM asks for 16385 MiB but this host has 16384 MiB");
    }

    #[test]
    fn session_hours_defaults_to_24_and_is_refused_outside_1_to_720() {
        let p = |l: &str| parse(&format!(r#"{{ "provider": {{ "models": ["m"] }}, "limits": {l} }}"#));
        assert_eq!(p("{}").unwrap().limits.session_hours, 24);
        assert_eq!(Limits::default().session_hours, 24);
        assert_eq!(p(r#"{ "sessionHours": 1 }"#).unwrap().limits.session_hours, 1);
        assert_eq!(p(r#"{ "sessionHours": 720 }"#).unwrap().limits.session_hours, 720);
        for bad in ["0", "721", "1.5", "\"24\"", "-1", "4294967297"] {
            assert_eq!(
                p(&format!(r#"{{ "sessionHours": {bad} }}"#)).unwrap_err(),
                "config: limits.sessionHours must be an integer from 1 to 720",
                "{bad}"
            );
        }
    }

    #[test]
    fn network_allow_is_read_validated_and_defaults_to_none() {
        assert!(parse("{}").unwrap().network.is_empty());
        let c = parse(r#"{ "network": { "allow": ["packages", "192.168.1.20:8080"] } }"#).unwrap();
        assert_eq!(c.network, ["packages", "192.168.1.20:8080"]);
        assert!(
            parse(r#"{ "network": { "allow": ["127.0.0.1:8463"] } }"#)
                .unwrap_err()
                .starts_with("config: network.allow: ")
        );
        assert!(parse(r#"{ "network": { "allow": [], "peer": [] } }"#).is_err());
    }

    #[test]
    fn config_values_win_over_installed_defaults() {
        let installed = Defaults {
            plugin_dir: "/p/plugin".into(),
            sessions_dir: "/p/sessions".into(),
            opencode: "/p/opencode".into(),
            mvmctl: "/p/mvmctl".into(),
        };
        let c = parse_config(
            r#"{ "pluginDir": "pl", "sessionsDir": "s", "opencode": "/oc", "mvmctl": "/m", "provider": { "models": ["m"] } }"#,
            Path::new("/cfg"),
            &installed,
        )
        .unwrap();
        assert_eq!(c.plugin_dir, PathBuf::from("/cfg/pl"));
        assert_eq!(c.sessions_dir, PathBuf::from("/cfg/s"));
        assert_eq!(c.opencode, "/oc");
        assert_eq!(c.mvmctl, PathBuf::from("/m"));
        let c = parse_config("{}", Path::new("/cfg"), &installed).unwrap();
        assert_eq!(c.plugin_dir, PathBuf::from("/p/plugin"));
        assert_eq!(c.sessions_dir, PathBuf::from("/p/sessions"));
        assert_eq!(c.opencode, "/p/opencode");
        assert_eq!(c.mvmctl, PathBuf::from("/p/mvmctl"));
    }

    #[test]
    fn model_display_names_are_readable_and_never_change_the_id() {
        assert_eq!(model_display_name("deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated"), "DeepSeek V4.1 Flash Abliterated");
        assert_eq!(model_display_name("z-ai/glm-5.3-flash"), "glm-5.3-flash");
        assert_eq!(model_display_name("org/team/model-x"), "model-x");
        assert_eq!(model_display_name("bare-model"), "bare-model");
        // A trailing slash leaves nothing after it; show the whole ID.
        assert_eq!(model_display_name("odd/"), "odd/");
    }

    #[test]
    fn a_crafted_model_id_cannot_fake_a_footer_segment_or_send_escapes() {
        for id in ["x · attested", "evil\u{1b}]0;pwned\u{7}name", "a\u{2022}attested", "rtl\u{202e}detsetta", "tab\there\nnl"] {
            let n = model_display_name(id);
            assert!(!n.chars().any(|c| c.is_control() || matches!(c, '·' | '•' | '\u{202e}')), "{n:?}");
        }
        assert_eq!(model_display_name("x · attested"), "x attested");
        assert_eq!(model_display_name("org/evil\u{1b}]0;pwned\u{7}"), "evil ]0;pwned");
        assert_eq!(model_display_name(&"m".repeat(500)).chars().count(), 64);
        assert_eq!(model_display_name("org/\u{1b}\u{7}"), "unnamed model");
    }

    #[test]
    fn an_operator_config_cannot_claim_the_embedded_provider_id() {
        for id in ["nearai-red-embedded", "nearai-red-direct"] {
            for t in [
                format!(r#"{{ "provider": {{ "id": "{id}", "baseURL": "https://x.example/v1", "apiKeyEnv": "K" }} }}"#),
                format!(r#"{{ "provider": {{ "id": "{id}" }} }}"#),
            ] {
                let e = parse(&t).unwrap_err();
                assert!(e.contains(id) && e.contains("reserved"), "{e}");
            }
        }
    }

    #[test]
    fn the_inference_mode_follows_the_provider_id() {
        assert_eq!(inference_mode(EMBEDDED_PROVIDER_ID), "ironwire");
        assert_eq!(inference_mode(DIRECT_PROVIDER_ID), "direct");
        assert_eq!(inference_mode("ironwire"), "external");
        assert_eq!(inference_mode("nearai"), "external");
    }

    #[test]
    fn the_default_model_is_near_ais_abliterated_deepseek() {
        let c = parse("{}").unwrap();
        assert_eq!(
            c.provider.models,
            ["deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated"]
        );
        assert!(require_real_models(&c.provider.models).is_ok());
        for m in DEFAULT_MODELS {
            assert!(crate::inference::route_header(m).is_ok(), "{m}");
        }
    }

    #[test]
    fn a_config_that_still_names_the_retired_placeholder_is_refused() {
        let c = parse(&format!(
            r#"{{"provider": {{"models": ["{MODEL_ID_PLACEHOLDER}"]}}}}"#
        ))
        .unwrap();
        let e = require_real_models(&c.provider.models).unwrap_err();
        assert!(
            e.contains("placeholder") && e.contains("provider.models"),
            "{e}"
        );
        assert!(require_real_models(&["z-ai/glm-5.3-flash".into()]).is_ok());
        assert!(require_real_models(&["a".into(), MODEL_ID_PLACEHOLDER.into()]).is_err());
    }

    #[test]
    fn a_minimal_config_routes_inference_through_ironwire() {
        let c = parse(r#"{ "provider": { "models": ["z-ai/glm-5.3-flash"] } }"#).unwrap();
        assert!(c.embedded_inference);
        assert_eq!(c.provider.id, "ironwire");
        assert_eq!(c.provider.base_url, "http://127.0.0.1:8463/openai/v1");
        assert_eq!(c.provider.api_key_env, None);
        assert_eq!(
            c.provider.loopback(),
            Some(("127.0.0.1".into(), 8463, "/openai/v1".into()))
        );
        assert_eq!(c.image, DEFAULT_IMAGE);
        assert_eq!(c.limits, Limits::default());
        assert_eq!(c.sessions_dir, PathBuf::from("/repo/sessions"));
        assert_eq!(c.plugin_dir, PathBuf::from("/repo/plugin"));
        assert_eq!(c.mvmctl, d().mvmctl);
    }

    #[test]
    fn a_direct_https_provider_needs_a_key_variable() {
        let direct = r#"{ "provider": { "id": "nearai", "baseURL": "https://cloud-api.near.ai/v1", "models": ["m"] } }"#;
        assert!(parse(direct).unwrap_err().contains("apiKeyEnv is required"));
        let keyed = r#"{ "provider": { "id": "nearai", "baseURL": "https://cloud-api.near.ai/v1", "models": ["m"], "apiKeyEnv": "NEARAI_API_KEY" } }"#;
        assert_eq!(
            parse(keyed).unwrap().provider.api_key_env.as_deref(),
            Some("NEARAI_API_KEY")
        );
        assert!(!parse(keyed).unwrap().embedded_inference);
    }

    #[test]
    fn an_explicit_external_loopback_provider_keeps_its_connection() {
        let cfg = parse(
            r#"{"provider":{"baseURL":"http://127.0.0.1:9999/openai/v1","models":["external"]}}"#,
        )
        .unwrap();
        assert!(!cfg.embedded_inference);
        assert_eq!(cfg.provider.base_url, "http://127.0.0.1:9999/openai/v1");
        assert_eq!(cfg.provider.models, ["external"]);
    }

    #[test]
    fn provider_credential_names_cannot_override_session_isolation_or_runtime() {
        for name in [
            "HOME",
            "PATH",
            "TMPDIR",
            "XDG_CONFIG_HOME",
            "OPENCODE_CONFIG_CONTENT",
            "RTS_TEST_LEAK",
            "LC_ALL",
            "LANG",
            "GIT_CONFIG_GLOBAL",
            "BASH_ENV",
            "ENV",
            "NODE_OPTIONS",
            "BUN_OPTIONS",
            "LD_PRELOAD",
            "DYLD_INSERT_LIBRARIES",
            "HTTPS_PROXY",
            "SSL_CERT_FILE",
        ] {
            let text = format!(
                r#"{{"provider":{{"id":"external","baseURL":"https://example.invalid/v1","apiKeyEnv":"{name}"}}}}"#
            );
            assert!(
                parse(&text).is_err(),
                "accepted reserved credential name {name}"
            );
        }
        for name in [
            "NEARAI_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "MY_PROVIDER_TOKEN",
            "K",
        ] {
            let text = format!(
                r#"{{"provider":{{"id":"external","baseURL":"https://example.invalid/v1","apiKeyEnv":"{name}"}}}}"#
            );
            assert!(parse(&text).is_ok(), "rejected credential name {name}");
        }
    }

    #[test]
    fn plain_http_is_refused_off_loopback() {
        for url in [
            "http://10.0.0.5:8463/openai/v1",
            "http://127.0.0.1/openai/v1",
            "http://127.0.0.1.evil:80/v1",
            "ftp://x",
        ] {
            let t = format!(
                r#"{{ "provider": {{ "baseURL": "{url}", "models": ["m"], "apiKeyEnv": "K" }} }}"#
            );
            assert!(parse(&t).unwrap_err().contains("baseURL"), "{url}");
        }
    }

    #[test]
    fn refusals_for_unpinned_images_bad_limits_and_unknown_keys() {
        assert!(
            parse(r#"{ "image": "alpine:latest", "provider": { "models": ["m"] } }"#)
                .unwrap_err()
                .contains("digest-pinned")
        );
        assert!(
            parse(r#"{ "provider": { "models": ["m"] }, "limits": { "cpus": 0 } }"#)
                .unwrap_err()
                .contains("limits.cpus")
        );
        assert!(
            parse(r#"{ "provider": { "models": [] } }"#)
                .unwrap_err()
                .contains("models")
        );
        assert!(
            parse(r#"{ "provider": { "models": ["m"] }, "mcp": {} }"#)
                .unwrap_err()
                .contains("unknown field")
        );
        assert!(
            parse(r#"{ "provider": { "models": ["m"], "apiKeyEnv": "lower" } }"#)
                .unwrap_err()
                .contains("apiKeyEnv")
        );
        assert!(
            parse(r#"{ "provider": { "id": "Bad_ID", "models": ["m"] } }"#)
                .unwrap_err()
                .contains("provider.id")
        );
    }

    #[test]
    fn relative_paths_resolve_against_the_config_file() {
        let c = parse(
            r#"{ "inputsDir": "./inputs", "sessionsDir": "out", "provider": { "models": ["m"] } }"#,
        )
        .unwrap();
        assert_eq!(c.inputs_dir, Some(PathBuf::from("/cfg/./inputs")));
        assert_eq!(c.sessions_dir, PathBuf::from("/cfg/out"));
    }

    #[test]
    fn the_example_config_parses() {
        let file = Path::new(env!("CARGO_MANIFEST_DIR")).join("opencode/session.example.json");
        let c = load_config(&file, &d()).unwrap();
        assert_eq!(c.provider.id, "ironwire");
        assert!(c.provider.loopback().is_some());
        // Copying the example and adding --memory must not trip a pinned ceiling.
        let mut l = c.limits;
        l.resize(None, Some(8192)).unwrap();
        assert_eq!(l.host_rss_ceiling_mib, 8192 + RSS_HEADROOM_MIB);
    }
}
