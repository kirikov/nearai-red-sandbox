//! Loopback forwarder between embedded IronWire and NEAR AI.
//!
//! IronWire's NEAR AI endpoint is this forwarder. It forwards only
//! `POST /v1/chat/completions` for allowlisted models, byte for byte, to
//! NEAR AI over TLS pinned to the attested certificate fingerprints
//! (`pin.rs`). A pin failure or a refused model raises `Fatal`, after which
//! nothing reaches upstream and the session loop ends the session.
//!
//! Each completion is hashed as it is relayed and, once it has ended, its
//! NEAR AI receipt is fetched and checked off the response path
//! (`signature.rs`). A failed check is fatal (`signature-failed`); every
//! completion leaves one line in `inference.jsonl`.

pub mod pin;
pub mod signature;

use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock, RwLock};
use std::task::{Context, Poll};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use http_body_util::combinators::BoxBody;
use http_body_util::{BodyExt, Full, LengthLimitError, Limited};
use hyper::body::{Frame, Incoming};
use hyper::header::{
    ACCEPT, ACCEPT_ENCODING, AUTHORIZATION, CONTENT_ENCODING, CONTENT_TYPE, HeaderValue,
};
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use sha2::{Digest, Sha256};
use tokio::sync::oneshot;

use crate::attest::Attested;
use crate::attest::record::AttestationRecord;
use pin::{PinSet, Rejected};
use signature::{Exchange, FetchError, InferenceLine, SignaturePolicy, SseId};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);
const RUNTIME_STOP_TIMEOUT: Duration = Duration::from_secs(1);
/// Beyond the drain of pending checks (up to the signature window).
const WORKER_STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// How often the shutdown drain looks for pending checks.
const DRAIN_POLL: Duration = Duration::from_millis(20);
const REATTEST_ATTEMPTS: usize = 3;
/// Per read on the upstream connection, resetting after each read: the
/// default `upstream_timeout_secs` IronWire bf8b90b applies to its own reads
/// (`ironwire_core/src/config.rs`), so the forwarder never gives up on a
/// response IronWire still waits for, and a stalled upstream is not read
/// forever after IronWire has gone. Before the response head this is a 502;
/// mid-body it is a stream error, which is fatal.
const UPSTREAM_READ_TIMEOUT: Duration = Duration::from_secs(900);
/// How often a relay or a signature fetch looks for a fatal raised elsewhere.
const FATAL_POLL: Duration = Duration::from_millis(50);

/// The first fatal reason; set once, read by the session loop.
pub struct Fatal {
    reason: OnceLock<String>,
}

impl std::fmt::Debug for Fatal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fatal")
            .field("reason", &self.reason.get())
            .finish()
    }
}

impl Fatal {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            reason: OnceLock::new(),
        })
    }

    /// Keeps the first reason; later ones are dropped. True when this call
    /// set it.
    pub fn raise(&self, reason: impl Into<String>) -> bool {
        self.reason.set(reason.into()).is_ok()
    }

    pub fn reason(&self) -> Option<String> {
        self.reason.get().cloned()
    }
}

/// Re-attests NEAR AI. Blocking (`BlockingHttp`): the forwarder calls it with
/// `tokio::task::spawn_blocking`.
pub type Reattest = Arc<dyn Fn() -> Result<Attested, String> + Send + Sync>;

/// Holds `reattest`, which captures the API key, so it has no `Debug`.
pub struct ForwarderConfig {
    /// "https://cloud-api.near.ai/v1"; tests use https://127.0.0.1:<port>/v1
    pub upstream: String,
    pub models: Vec<String>,
    pub roots: Arc<rustls::RootCertStore>,
    pub seed: Attested,
    /// Called with tokio::task::spawn_blocking.
    pub reattest: Reattest,
    pub fatal: Arc<Fatal>,
    pub record: Arc<AttestationRecord>,
    /// 32 MiB in production.
    pub max_request_bytes: usize,
    /// Pin the upstream TLS certificate to the attested fingerprints (the
    /// launch path). `false` uses ordinary TLS to the host and attests by the
    /// per-response signatures only, so the gateway can fail over (the NEAR AI
    /// documented verification path).
    pub pin_tls: bool,
    pub signatures: signature::SignaturePolicy,
    /// `record.session_dir().join("inference.jsonl")`.
    pub inference_log: std::path::PathBuf,
}

/// Owns the loopback listener and its runtime thread, as `EmbeddedInference`.
pub struct Forwarder {
    port: u16,
    /// The signature window: how long `drop` lets pending checks finish.
    drain: Duration,
    shutdown: Option<oneshot::Sender<()>>,
    finished: mpsc::Receiver<()>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for Forwarder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Forwarder")
            .field("port", &self.port)
            .finish_non_exhaustive()
    }
}

struct State {
    completions: String,
    models: BTreeSet<String>,
    client: reqwest::Client,
    pins: Arc<PinSet>,
    reattest: Reattest,
    reattesting: tokio::sync::Mutex<()>,
    fatal: Arc<Fatal>,
    record: Arc<AttestationRecord>,
    max_request_bytes: usize,
    /// "https://cloud-api.near.ai/v1", for `/signature/{id}`.
    upstream: String,
    /// The seed's attested signing keys per model, merged with every
    /// re-attestation.
    keys: RwLock<BTreeMap<String, BTreeSet<[u8; 32]>>>,
    /// The attested gateways' ed25519 keys, valid receipt signers for any
    /// allowed model; merged with every re-attestation like `keys`.
    gateway_keys: RwLock<BTreeSet<[u8; 32]>>,
    signatures: SignaturePolicy,
    inference_log: PathBuf,
    /// Responses relayed or being checked that have no line yet. Whoever
    /// removes an entry writes its line, under this lock: the check task, or
    /// the shutdown drain for what is still here when the window ends. The
    /// lock also keeps lines from interleaving.
    pending: Mutex<BTreeMap<u64, Open>>,
    next_check: std::sync::atomic::AtomicU64,
}

/// What is known of a response before its check settles.
struct Open {
    model: String,
    streamed: bool,
    request_sha256: [u8; 32],
    chat_id: String,
    /// Set once the relay has read the whole response.
    response_sha256: Option<[u8; 32]>,
}

type Out = Response<BoxBody<Bytes, BoxError>>;
type BoxError = Box<dyn std::error::Error + Send + Sync>;

impl Forwarder {
    pub fn start(cfg: ForwarderConfig) -> Result<Forwarder, String> {
        if cfg.pin_tls && cfg.seed.fingerprints.is_empty() {
            return Err("forwarder: no attested certificate fingerprints to pin".into());
        }
        if !cfg.pin_tls && cfg.seed.keys.is_empty() {
            // Signature mode has no TLS pin; the per-response signatures are the
            // only attestation binding, so the attested model keys are required.
            return Err("forwarder: signature mode needs attested model signing keys".into());
        }
        if cfg.models.is_empty() {
            return Err("forwarder: no allowed models".into());
        }
        let upstream = cfg.upstream.trim_end_matches('/');
        if !upstream.starts_with("https://") {
            return Err("forwarder: upstream must be https".into());
        }
        let drain = cfg.signatures.window;
        let pins = PinSet::new(cfg.seed.fingerprints.clone());
        // Pinned mode binds the connection to the attested instance's cert;
        // signature mode uses ordinary TLS to the host and relies on the
        // per-response signatures (which survive gateway failover).
        let tls = if cfg.pin_tls {
            pin::pinned_tls(cfg.roots, pins.clone())?
        } else {
            pin::native_tls(cfg.roots)?
        };
        let client = reqwest::Client::builder()
            .use_preconfigured_tls(tls)
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .http1_only()
            .no_gzip()
            .no_brotli()
            .no_deflate()
            .no_zstd()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(UPSTREAM_READ_TIMEOUT)
            .build()
            .map_err(|e| format!("forwarder: upstream client: {e}"))?;
        let state = Arc::new(State {
            completions: format!("{upstream}/chat/completions"),
            models: cfg.models.into_iter().collect(),
            client,
            pins,
            reattest: cfg.reattest,
            reattesting: tokio::sync::Mutex::new(()),
            fatal: cfg.fatal,
            record: cfg.record,
            max_request_bytes: cfg.max_request_bytes,
            upstream: upstream.to_string(),
            keys: RwLock::new(cfg.seed.keys),
            gateway_keys: RwLock::new(cfg.seed.gateway_keys),
            signatures: cfg.signatures,
            inference_log: cfg.inference_log,
            pending: Mutex::new(BTreeMap::new()),
            next_check: std::sync::atomic::AtomicU64::new(0),
        });
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0))
            .map_err(|e| format!("forwarder: cannot bind 127.0.0.1: {e}"))?;
        let port = listener
            .local_addr()
            .map_err(|e| format!("forwarder: {e}"))?
            .port();
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("forwarder: {e}"))?;
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let (done_tx, finished) = mpsc::sync_channel(1);
        let (shutdown, shutdown_rx) = oneshot::channel();
        let worker = std::thread::Builder::new()
            .name("rts-forwarder".into())
            .spawn(move || {
                run_worker(listener, state, ready_tx, shutdown_rx);
                let _ = done_tx.send(());
            })
            .map_err(|_| "forwarder: cannot start worker thread")?;
        match ready_rx.recv_timeout(STARTUP_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err("forwarder: startup did not complete".into()),
        }
        Ok(Forwarder {
            port,
            drain,
            shutdown: Some(shutdown),
            finished,
            worker: Some(worker),
        })
    }

    /// "http://127.0.0.1:<port>/v1"
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }
}

impl Drop for Forwarder {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if self
            .finished
            .recv_timeout(self.drain + WORKER_STOP_TIMEOUT)
            .is_ok()
            && let Some(worker) = self.worker.take()
        {
            let _ = worker.join();
        }
    }
}

fn run_worker(
    listener: std::net::TcpListener,
    state: Arc<State>,
    ready: mpsc::SyncSender<Result<(), String>>,
    shutdown: oneshot::Receiver<()>,
) {
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(_) => {
            let _ = ready.send(Err("forwarder: cannot initialize runtime".into()));
            return;
        }
    };
    runtime.block_on(async {
        let listener = match tokio::net::TcpListener::from_std(listener) {
            Ok(listener) => listener,
            Err(e) => {
                let _ = ready.send(Err(format!("forwarder: listener: {e}")));
                return;
            }
        };
        let accept = tokio::spawn(accept_loop(listener, state.clone()));
        if ready.send(Ok(())).is_ok() {
            let _ = shutdown.await;
        }
        accept.abort();
        drain(&state).await;
    });
    runtime.shutdown_timeout(RUNTIME_STOP_TIMEOUT);
}

/// Lets pending checks finish for up to the signature window; then each one
/// still pending gets a `failed: unchecked at shutdown` line and the session
/// is marked `signature-failed`, so nothing is left looking verified.
async fn drain(state: &State) {
    let deadline = tokio::time::Instant::now() + state.signatures.window;
    while tokio::time::Instant::now() < deadline && !lock(&state.pending).is_empty() {
        tokio::time::sleep(DRAIN_POLL).await;
    }
    let mut pending = lock(&state.pending);
    if pending.is_empty() {
        return;
    }
    let reason = "unchecked at shutdown";
    let mut written = Ok(());
    for open in std::mem::take(&mut *pending).into_values() {
        let line = InferenceLine {
            ts: now_secs(),
            chat_id: open.chat_id,
            model: open.model,
            streamed: open.streamed,
            request_sha256: hex::encode(open.request_sha256),
            response_sha256: open.response_sha256.map(hex::encode).unwrap_or_default(),
            signing_key: None,
            signer: None,
            signature_ms: None,
            result: format!("failed: {reason}"),
        };
        written = written.and(write_line(state, &line));
    }
    stop(&state.fatal, &state.record, "signature-failed", reason);
    unwritten(state, written);
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Appends one line. The caller holds `pending`, and passes the result to
/// `unwritten` after recording any failure of its own, so that failure owns
/// the status.
fn write_line(state: &State, line: &InferenceLine) -> std::io::Result<()> {
    signature::append_inference_line(&state.inference_log, line)
}

/// A line that cannot be written is fatal.
fn unwritten(state: &State, written: std::io::Result<()>) {
    if let Err(e) = written {
        // The kind only: never a path or a header.
        eprintln!("[rts forwarder] cannot write inference.jsonl: {}", e.kind());
        stop(
            &state.fatal,
            &state.record,
            "signature-failed",
            "cannot write inference.jsonl",
        );
    }
}

async fn accept_loop(listener: tokio::net::TcpListener, state: Arc<State>) {
    loop {
        let stream = match listener.accept().await {
            Ok((stream, _)) => stream,
            Err(_) => {
                tokio::time::sleep(Duration::from_millis(10)).await;
                continue;
            }
        };
        let state = state.clone();
        tokio::spawn(async move {
            let service = hyper::service::service_fn(move |req| {
                let state = state.clone();
                async move { Ok::<_, Infallible>(handle(&state, req).await) }
            });
            let _ = hyper::server::conn::http1::Builder::new()
                .serve_connection(TokioIo::new(stream), service)
                .await;
        });
    }
}

fn local(status: StatusCode, message: &'static str) -> Out {
    let mut resp = Response::new(
        Full::new(Bytes::from_static(message.as_bytes()))
            .map_err(|never| match never {})
            .boxed(),
    );
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static("text/plain"));
    resp
}

/// Raises fatal and, only if this is the first fatal, writes the matching
/// terminal status, so attestation.json and the stop reason agree.
fn stop(fatal: &Fatal, record: &AttestationRecord, status: &str, reason: &str) -> bool {
    let first = fatal.raise(format!("{status}: {reason}"));
    if first {
        record.set_status(status, reason);
    }
    first
}

/// Method and path only: never headers, query or body.
fn not_forwarded_line(method: &Method, path: &str) -> String {
    format!("[rts forwarder] 404 {method} {path}: not forwarded")
}

fn stopped_reply() -> Out {
    local(
        StatusCode::BAD_GATEWAY,
        "rts forwarder: inference stopped; see attestation.json",
    )
}

fn fatal_reply(state: &State, status: &str, reason: String) -> Out {
    stop(&state.fatal, &state.record, status, &reason);
    stopped_reply()
}

async fn handle(state: &Arc<State>, req: Request<Incoming>) -> Out {
    // Fatal is checked here, again once the body has arrived, before every
    // upstream send, and inside `learn`: it can be raised at any point.
    if state.fatal.reason().is_some() {
        return stopped_reply();
    }
    if req.method() != Method::POST || req.uri().path() != "/v1/chat/completions" {
        eprintln!("{}", not_forwarded_line(req.method(), req.uri().path()));
        return local(StatusCode::NOT_FOUND, "rts forwarder: not forwarded");
    }
    let mut headers = reqwest::header::HeaderMap::new();
    for name in [AUTHORIZATION, CONTENT_TYPE, ACCEPT] {
        for value in req.headers().get_all(&name) {
            headers.append(name.clone(), value.clone());
        }
    }
    headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));
    let body = match Limited::new(req.into_body(), state.max_request_bytes)
        .collect()
        .await
    {
        Ok(body) => body.to_bytes(),
        Err(e) if e.downcast_ref::<LengthLimitError>().is_some() => {
            return local(
                StatusCode::PAYLOAD_TOO_LARGE,
                "rts forwarder: request too large",
            );
        }
        Err(_) => return local(StatusCode::BAD_REQUEST, "rts forwarder: unreadable request"),
    };
    if state.fatal.reason().is_some() {
        return stopped_reply();
    }
    let Some(model) = request_model(&body) else {
        return local(
            StatusCode::BAD_REQUEST,
            "rts forwarder: body must be a JSON object with one string \"model\"",
        );
    };
    if !state.models.contains(&model) {
        let shown: String = model.chars().take(128).collect();
        stop(
            &state.fatal,
            &state.record,
            "model-refused",
            &format!("model {shown:?} is not in the allowlist"),
        );
        return local(
            StatusCode::FORBIDDEN,
            "rts forwarder: model is not in the allowlist",
        );
    }
    // The exact bytes received, never a re-serialization; hashed as sent.
    let sent = Sent {
        model,
        request_sha256: Sha256::digest(&body).into(),
        authorization: headers.get(AUTHORIZATION).cloned().map(|mut v| {
            v.set_sensitive(true);
            v
        }),
    };
    let send = || {
        state
            .client
            .post(&state.completions)
            .headers(headers.clone())
            .body(body.clone())
            .send()
    };
    if state.fatal.reason().is_some() {
        return stopped_reply();
    }
    let resp = match send().await {
        Ok(resp) => resp,
        Err(_) => match state.pins.take_rejected() {
            None => {
                return local(
                    StatusCode::BAD_GATEWAY,
                    "rts forwarder: upstream request failed",
                );
            }
            Some(r @ Rejected::Untrusted(_)) => {
                return fatal_reply(
                    state,
                    "pin-failed",
                    format!("pin: {} does not chain to a trusted root", r.certificate()),
                );
            }
            Some(Rejected::Unpinned(fp)) => {
                match learn(state, fp).await {
                    Ok(()) => {}
                    Err(Unlearned::Stopped) => return stopped_reply(),
                    Err(Unlearned::Failed(reason)) => {
                        return fatal_reply(state, "pin-failed", reason);
                    }
                }
                if state.fatal.reason().is_some() {
                    return stopped_reply();
                }
                match send().await {
                    Ok(resp) => resp,
                    Err(_) => {
                        if let Some(r) = state.pins.take_rejected() {
                            return fatal_reply(
                                state,
                                "pin-failed",
                                format!("pin: {} refused after re-attestation", r.certificate()),
                            );
                        }
                        return local(
                            StatusCode::BAD_GATEWAY,
                            "rts forwarder: upstream request failed",
                        );
                    }
                }
            }
        },
    };
    relay(state.clone(), resp, sent)
}

enum Unlearned {
    /// Fatal was raised meanwhile (possibly by another request's `learn`).
    Stopped,
    Failed(String),
}

/// Re-attests (on the blocking pool, one at a time) until `fp` is attested,
/// recording each re-attestation. Stops as soon as fatal is raised: after
/// taking the lock, before each attempt, and before recording a result.
async fn learn(state: &State, fp: [u8; 32]) -> Result<(), Unlearned> {
    let stopped = || state.fatal.reason().is_some();
    let _one_at_a_time = state.reattesting.lock().await;
    for _ in 0..REATTEST_ATTEMPTS {
        if stopped() {
            return Err(Unlearned::Stopped);
        }
        if state.pins.contains(&fp) {
            return Ok(());
        }
        reattest_once(state, "pin").await?;
    }
    if stopped() {
        return Err(Unlearned::Stopped);
    }
    if state.pins.contains(&fp) {
        return Ok(());
    }
    Err(Unlearned::Failed(format!(
        "pin: upstream certificate with public key sha256 {} is not attested after {REATTEST_ATTEMPTS} re-attestations",
        hex::encode(fp)
    )))
}

/// One re-attestation, by a caller holding `reattesting`: run on the
/// blocking pool, recorded, and merged into the pins and the keys. Stops if
/// fatal is raised before it starts or before its result is recorded.
async fn reattest_once(state: &State, what: &str) -> Result<(), Unlearned> {
    let stopped = || state.fatal.reason().is_some();
    if stopped() {
        return Err(Unlearned::Stopped);
    }
    let reattest = state.reattest.clone();
    let attested = tokio::task::spawn_blocking(move || reattest())
        .await
        .map_err(|_| Unlearned::Failed(format!("{what}: re-attestation did not complete")))?
        .map_err(|e| Unlearned::Failed(format!("{what}: re-attestation failed: {e}")))?;
    if stopped() {
        return Err(Unlearned::Stopped);
    }
    state
        .record
        .reattested(&attested)
        .map_err(|e| Unlearned::Failed(format!("{what}: cannot record re-attestation: {e}")))?;
    state.pins.extend(&attested.fingerprints);
    let mut keys = state.keys.write().unwrap_or_else(|e| e.into_inner());
    for (model, more) in attested.keys {
        keys.entry(model).or_default().extend(more);
    }
    drop(keys);
    state
        .gateway_keys
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .extend(attested.gateway_keys);
    Ok(())
}

/// What `handle` sent upstream, for the receipt check.
struct Sent {
    model: String,
    request_sha256: [u8; 32],
    /// The inbound Authorization, passed to the signature fetch. Marked
    /// sensitive; never formatted.
    authorization: Option<HeaderValue>,
}

/// How the relay of one response ended.
enum Relayed {
    Complete,
    /// Fatal was raised elsewhere while it was relaying.
    Stopped,
    Failed(String),
}

/// Resolves once fatal has been raised.
async fn fatal_raised(fatal: &Fatal) {
    while fatal.reason().is_none() {
        tokio::time::sleep(FATAL_POLL).await;
    }
}

/// True unless `Content-Encoding` is absent or only `identity`. The receipt
/// covers the bytes NEAR AI sent, which the forwarder must hash as received;
/// it asks for `identity`, so any other coding fails closed.
fn is_encoded(headers: &reqwest::header::HeaderMap) -> bool {
    headers
        .get_all(CONTENT_ENCODING)
        .iter()
        .any(|v| match v.to_str() {
            Err(_) => true,
            Ok(v) => v
                .split(',')
                .any(|coding| !coding.trim().eq_ignore_ascii_case("identity")),
        })
}

fn is_event_stream(content_type: Option<&HeaderValue>) -> bool {
    content_type
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(';').next())
        .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"))
}

/// Status, Content-Type and the body as it arrives. A task reads upstream to
/// the end, hashing every chunk it relays, even after the client has gone;
/// then checks the receipt. A fatal raised elsewhere cuts the relay.
fn relay(state: Arc<State>, mut upstream: reqwest::Response, sent: Sent) -> Out {
    let status = upstream.status();
    let content_type = upstream.headers().get(CONTENT_TYPE).cloned();
    let streamed = is_event_stream(content_type.as_ref());
    let encoded = is_encoded(upstream.headers());
    let id = state
        .next_check
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    lock(&state.pending).insert(
        id,
        Open {
            model: sent.model.clone(),
            streamed,
            request_sha256: sent.request_sha256,
            chat_id: String::new(),
            response_sha256: None,
        },
    );
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        let mut hasher = Sha256::new();
        let mut sse = SseId::default();
        // A non-streamed 2xx body, kept to find its chat ID.
        let mut kept = Vec::new();
        let keep = !streamed && status.is_success();
        let mut client = Some(tx);
        let relayed = if encoded {
            // Nothing is relayed: the body is not what the receipt covers.
            Relayed::Failed("upstream sent a Content-Encoding other than identity".into())
        } else {
            loop {
                let next = tokio::select! {
                    biased;
                    () = fatal_raised(&state.fatal) => break Relayed::Stopped,
                    next = upstream.chunk() => next,
                };
                let chunk = match next {
                    Ok(Some(chunk)) => chunk,
                    Ok(None) => break Relayed::Complete,
                    Err(_) => break Relayed::Failed("upstream stream failed".into()),
                };
                if keep {
                    if kept.len() + chunk.len() > signature::MAX_ID_SCAN_BYTES {
                        break Relayed::Failed(format!(
                            "response over {} bytes",
                            signature::MAX_ID_SCAN_BYTES
                        ));
                    }
                    kept.extend_from_slice(&chunk);
                } else if streamed {
                    sse.push(&chunk);
                }
                hasher.update(&chunk);
                if let Some(tx) = &client
                    && tx.send(Ok(chunk)).await.is_err()
                {
                    // The client went away: keep reading so the hash covers the
                    // whole response.
                    client = None;
                }
            }
        };
        if let Some(tx) = client {
            if !matches!(relayed, Relayed::Complete) {
                let _ = tx.send(Err("upstream response stopped".into())).await;
            }
            // Dropping the sender ends the client's body before the check.
            drop(tx);
        }
        let chat_id = if streamed {
            sse.id().map(str::to_string)
        } else {
            signature::chat_id_from_json(&kept)
        };
        let ex = Exchange {
            model: sent.model,
            chat_id: chat_id
                .filter(|id| signature::usable_id(id))
                .unwrap_or_default(),
            streamed,
            request_sha256: sent.request_sha256,
            response_sha256: hasher.finalize().into(),
        };
        if let Some(open) = lock(&state.pending).get_mut(&id) {
            open.chat_id = ex.chat_id.clone();
            open.response_sha256 = Some(ex.response_sha256);
        }
        check(&state, id, status, relayed, ex, sent.authorization).await;
    });
    let mut resp = Response::new(Relay { rx }.boxed());
    *resp.status_mut() = status;
    if let Some(content_type) = content_type {
        resp.headers_mut().insert(CONTENT_TYPE, content_type);
    }
    resp
}

/// Settles one relayed response: an `inference.jsonl` line and, on any
/// failure of a 2xx answer, `signature-failed`.
async fn check(
    state: &State,
    id: u64,
    status: StatusCode,
    relayed: Relayed,
    ex: Exchange,
    authorization: Option<HeaderValue>,
) {
    let outcome = match relayed {
        Relayed::Stopped => Checked::Stopped,
        _ if !status.is_success() => Checked::NotSigned,
        Relayed::Failed(reason) => Checked::Failed(reason),
        Relayed::Complete if ex.chat_id.is_empty() => {
            Checked::Failed("no usable chat ID in the response".into())
        }
        Relayed::Complete => match authorization {
            None => Checked::Failed("no Authorization to fetch the signature with".into()),
            Some(auth) => tokio::select! {
                biased;
                () = fatal_raised(&state.fatal) => Checked::Stopped,
                checked = verify(state, &ex, &auth) => checked,
            },
        },
    };
    let (result, signing_key, signer, signature_ms) = match &outcome {
        Checked::Verified(key, signer, took) => (
            "verified".to_string(),
            Some(hex::encode(key)),
            Some(signer.as_str()),
            Some(u64::try_from(took.as_millis()).unwrap_or(u64::MAX)),
        ),
        Checked::NotSigned => (
            format!("not-signed (HTTP {})", status.as_u16()),
            None,
            None,
            None,
        ),
        Checked::Stopped => ("failed: session stopped".to_string(), None, None, None),
        Checked::Failed(reason) | Checked::PinFailed(reason) => {
            (format!("failed: {reason}"), None, None, None)
        }
    };
    let line = InferenceLine {
        ts: now_secs(),
        chat_id: ex.chat_id.clone(),
        model: ex.model.clone(),
        streamed: ex.streamed,
        request_sha256: hex::encode(ex.request_sha256),
        response_sha256: hex::encode(ex.response_sha256),
        signing_key,
        signer,
        signature_ms,
        result,
    };
    let mut pending = lock(&state.pending);
    if pending.remove(&id).is_none() {
        // The shutdown drain gave up on it and wrote its line.
        return;
    }
    let written = write_line(state, &line);
    match &outcome {
        Checked::Failed(reason) => {
            stop(&state.fatal, &state.record, "signature-failed", reason);
        }
        Checked::PinFailed(reason) => {
            stop(&state.fatal, &state.record, "pin-failed", reason);
        }
        _ => {}
    }
    unwritten(state, written);
    drop(pending);
}

enum Checked {
    Verified([u8; 32], Signer, Duration),
    NotSigned,
    Stopped,
    Failed(String),
    /// Upstream's certificate was refused during the receipt fetch.
    PinFailed(String),
}

/// Which attested role's key signed a receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Signer {
    Model,
    Gateway,
}

impl Signer {
    fn as_str(self) -> &'static str {
        match self {
            Signer::Model => "model",
            Signer::Gateway => "gateway",
        }
    }
}

/// One snapshot of the keys that may sign a receipt for `model`: its own
/// attested keys and the attested gateway keys.
struct Signers {
    model: BTreeSet<[u8; 32]>,
    gateway: BTreeSet<[u8; 32]>,
}

impl Signers {
    fn all(&self) -> BTreeSet<[u8; 32]> {
        self.model.union(&self.gateway).copied().collect()
    }

    fn contains(&self, key: &[u8; 32]) -> bool {
        self.model.contains(key) || self.gateway.contains(key)
    }

    /// A key attested both ways is logged as the model's.
    fn role(&self, key: &[u8; 32]) -> Signer {
        if self.model.contains(key) {
            Signer::Model
        } else {
            Signer::Gateway
        }
    }
}

fn attested_keys(state: &State, model: &str) -> Signers {
    Signers {
        model: state
            .keys
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .get(model)
            .cloned()
            .unwrap_or_default(),
        gateway: state
            .gateway_keys
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone(),
    }
}

/// Fetches and checks the receipt. A certificate the fetch sees refused is
/// handled as one a completion sees: an unpinned one is learned (`learn`,
/// at most once per fetch) and the fetch retried within what is left of the
/// window; an untrusted one, or one still refused, is `pin-failed`. A key
/// attested neither for the model nor for the gateway gets one
/// re-attestation, then the check runs again.
async fn verify(state: &State, ex: &Exchange, auth: &HeaderValue) -> Checked {
    let start = Instant::now();
    let deadline = start + state.signatures.window;
    let mut learned = false;
    let (receipt, took) = loop {
        let policy = SignaturePolicy {
            window: deadline.saturating_duration_since(Instant::now()),
            ..state.signatures
        };
        if policy.window.is_zero() {
            return Checked::Failed(format!(
                "signature: no signature within {} ms (re-attested for a rotated certificate)",
                state.signatures.window.as_millis()
            ));
        }
        let fetched = signature::fetch_receipt(
            &state.client,
            &state.upstream,
            auth,
            ex,
            policy,
            &state.pins,
        )
        .await;
        match fetched {
            // The fetch that answered: signature_ms stays the fetch time.
            Ok(fetched) => break fetched,
            Err(FetchError::Failed(e)) => return Checked::Failed(format!("signature: {e}")),
            Err(FetchError::Rejected(r @ Rejected::Untrusted(_))) => {
                return Checked::PinFailed(format!(
                    "pin: {} does not chain to a trusted root",
                    r.certificate()
                ));
            }
            Err(FetchError::Rejected(r @ Rejected::Unpinned(_))) if learned => {
                return Checked::PinFailed(format!(
                    "pin: {} refused after re-attestation",
                    r.certificate()
                ));
            }
            Err(FetchError::Rejected(Rejected::Unpinned(fp))) => {
                learned = true;
                match learn(state, fp).await {
                    Ok(()) => {}
                    Err(Unlearned::Stopped) => return Checked::Stopped,
                    Err(Unlearned::Failed(reason)) => return Checked::PinFailed(reason),
                }
            }
        }
    };
    if let Ok(key) = signature::receipt_key(&receipt)
        && !attested_keys(state, &ex.model).contains(&key)
    {
        let _one_at_a_time = state.reattesting.lock().await;
        // Another task may have learned it while this one waited.
        if !attested_keys(state, &ex.model).contains(&key) {
            match reattest_once(state, "signature").await {
                Ok(()) => {}
                Err(Unlearned::Stopped) => return Checked::Stopped,
                Err(Unlearned::Failed(reason)) => return Checked::Failed(reason),
            }
        }
    }
    // One snapshot for both the check and the role it logs.
    let signers = attested_keys(state, &ex.model);
    match signature::check_receipt(&receipt, ex, &signers.all()) {
        Ok(key) => Checked::Verified(key, signers.role(&key), took),
        Err(e) => Checked::Failed(format!("signature: {e}")),
    }
}

struct Relay {
    rx: tokio::sync::mpsc::Receiver<Result<Bytes, BoxError>>,
}

impl hyper::body::Body for Relay {
    type Data = Bytes;
    type Error = BoxError;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, BoxError>>> {
        self.rx
            .poll_recv(cx)
            .map(|chunk| chunk.map(|chunk| chunk.map(Frame::data)))
    }
}

/// The top-level string "model" of a JSON object. A second "model" key is
/// refused: parsers disagree on which one wins.
fn request_model(body: &[u8]) -> Option<String> {
    signature::top_level_string(body, "model")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attest::record::AttestationRecord;
    use crate::attest::{AttestOptions, Attested};
    use crate::forwarder::signature::SignaturePolicy;
    use ed25519_dalek::{Signer, SigningKey};
    use rustls::HandshakeKind;
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, PrivateKeyDer};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeSet;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    const DUMMY_KEY: &str = "rts-dummy-key-never-send-to-a-provider";
    const BODY: &[u8] = br#"{"model":"m","messages":[],"stream":false}"#;
    const REPLY: &[u8] = br#"{"id":"x","object":"chat.completion","choices":[]}"#;
    const STREAM_BODY: &[u8] = br#"{"model":"m","messages":[],"stream":true}"#;
    const SSE: [&[u8]; 3] = [
        b"data: {\"id\":\"chatcmpl-s1\",\"choices\":[{\"delta\":{\"content\":\"he\"}}]}\n\n",
        b"data: {\"id\":\"chatcmpl-s1\",\"choices\":[{\"delta\":{\"content\":\"llo\"}}]}\n\n",
        b"data: [DONE]\n\n",
    ];

    fn test_key() -> SigningKey {
        SigningKey::from_bytes(&[9; 32])
    }

    fn test_vk() -> [u8; 32] {
        test_key().verifying_key().to_bytes()
    }

    fn sha_hex(b: &[u8]) -> String {
        hex::encode(Sha256::digest(b))
    }

    /// The key NEAR AI production signs receipts with: the gateway's.
    fn gateway_key() -> SigningKey {
        SigningKey::from_bytes(&[10; 32])
    }

    fn gateway_vk() -> [u8; 32] {
        gateway_key().verifying_key().to_bytes()
    }

    /// A receipt as NEAR AI returns it, signed with the test key.
    fn receipt_for(text: &str) -> Vec<u8> {
        receipt_signed(&test_key(), text)
    }

    fn receipt_signed(key: &SigningKey, text: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "text": text,
            "signature": hex::encode(key.sign(text.as_bytes()).to_bytes()),
            "signing_address": hex::encode(key.verifying_key().to_bytes()),
            "signing_algo": "ed25519",
            "signature_kind": "provider_tee",
        }))
        .unwrap()
    }

    /// As `good_receipt`, signed with the gateway key.
    fn gateway_receipt(req: &[u8], resp: &[u8]) -> (u16, Vec<u8>) {
        (
            200,
            receipt_signed(
                &gateway_key(),
                &format!("m:{}:{}", sha_hex(req), sha_hex(resp)),
            ),
        )
    }

    /// Signs sha256(request body received) and sha256(response body sent).
    fn good_receipt(req: &[u8], resp: &[u8]) -> (u16, Vec<u8>) {
        (
            200,
            receipt_for(&format!("m:{}:{}", sha_hex(req), sha_hex(resp))),
        )
    }

    /// Signs another response hash.
    fn wrong_receipt(req: &[u8], _: &[u8]) -> (u16, Vec<u8>) {
        (
            200,
            receipt_for(&format!("m:{}:{}", sha_hex(req), sha_hex(b"other"))),
        )
    }

    fn quick() -> SignaturePolicy {
        SignaturePolicy {
            window: Duration::from_secs(2),
            first_backoff: Duration::from_millis(20),
            max_backoff: Duration::from_millis(100),
        }
    }

    /// What the test upstream answers to a completion.
    enum Answer {
        Whole(u16, Vec<u8>),
        /// 200 JSON with `Content-Encoding: <the str>`; the bytes as given.
        Encoded(&'static str, Vec<u8>),
        /// `text/event-stream`, chunked: `pauses[i]` before chunk i; the time
        /// just before the last chunk is written goes to `last_sent`.
        Sse {
            chunks: Vec<Vec<u8>>,
            pauses: Vec<Duration>,
            last_sent: Arc<Mutex<Option<Instant>>>,
        },
    }

    impl Answer {
        fn bytes(&self) -> Vec<u8> {
            match self {
                Answer::Whole(_, b) | Answer::Encoded(_, b) => b.clone(),
                Answer::Sse { chunks, .. } => chunks.concat(),
            }
        }
    }

    fn sse_answer(pauses: Vec<Duration>, last_sent: &Arc<Mutex<Option<Instant>>>) -> Answer {
        Answer::Sse {
            chunks: SSE.iter().map(|c| c.to_vec()).collect(),
            pauses,
            last_sent: last_sent.clone(),
        }
    }

    fn pem(name: &str) -> &'static [u8] {
        match name {
            "ca" => include_bytes!("../tests/fixtures/tls/ca.pem"),
            "a" => include_bytes!("../tests/fixtures/tls/a.pem"),
            "b" => include_bytes!("../tests/fixtures/tls/b.pem"),
            "c" => include_bytes!("../tests/fixtures/tls/c.pem"),
            "a.key" => include_bytes!("../tests/fixtures/tls/a.key"),
            "b.key" => include_bytes!("../tests/fixtures/tls/b.key"),
            "c.key" => include_bytes!("../tests/fixtures/tls/c.key"),
            _ => unreachable!(),
        }
    }

    fn cert(name: &str) -> CertificateDer<'static> {
        CertificateDer::from_pem_slice(pem(name)).unwrap()
    }

    fn fp(name: &str) -> [u8; 32] {
        pin::spki_sha256(&cert(name)).expect("parses")
    }

    fn fps(names: &[&str]) -> BTreeSet<[u8; 32]> {
        names.iter().map(|n| fp(n)).collect()
    }

    fn roots() -> Arc<rustls::RootCertStore> {
        let mut store = rustls::RootCertStore::empty();
        store.add(cert("ca")).unwrap();
        Arc::new(store)
    }

    fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    /// One recorded upstream request: the head (request line and headers,
    /// as sent) and the exact body bytes.
    #[derive(Clone)]
    struct Seen {
        head: String,
        body: Vec<u8>,
    }

    /// A blocking TLS upstream on 127.0.0.1. `certs(i)` names the
    /// certificate presented on the i-th accepted connection. Every accepted
    /// connection is recorded, with None when its handshake failed.
    struct Upstream {
        port: u16,
        handshakes: Arc<Mutex<Vec<Option<HandshakeKind>>>>,
        requests: Arc<Mutex<Vec<Seen>>>,
        signatures: Arc<Mutex<Vec<Seen>>>,
    }

    impl Upstream {
        fn url(&self) -> String {
            format!("https://127.0.0.1:{}/v1", self.port)
        }
        fn handshakes(&self) -> Vec<Option<HandshakeKind>> {
            self.handshakes.lock().unwrap().clone()
        }
        /// Completions only.
        fn requests(&self) -> Vec<Seen> {
            self.requests.lock().unwrap().clone()
        }
        fn signature_requests(&self) -> Vec<Seen> {
            self.signatures.lock().unwrap().clone()
        }
    }

    fn server_config(name: &'static str) -> Arc<rustls::ServerConfig> {
        let key = PrivateKeyDer::from_pem_slice(pem(&format!("{name}.key"))).unwrap();
        Arc::new(
            rustls::ServerConfig::builder_with_provider(Arc::new(
                rustls::crypto::ring::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![cert(name)], key)
            .unwrap(),
        )
    }

    /// Answers `GET /v1/signature/..` with a good receipt for the last
    /// completion.
    fn upstream(
        certs: impl Fn(usize) -> &'static str + Send + 'static,
        close_each: bool,
        reply: impl Fn(&Seen) -> Answer + Send + Sync + 'static,
    ) -> Upstream {
        upstream_with(certs, close_each, reply, good_receipt)
    }

    type Reply = dyn Fn(&Seen) -> Answer + Send + Sync;
    type Sign = dyn Fn(&[u8], &[u8]) -> (u16, Vec<u8>) + Send + Sync;
    type Last = Mutex<Option<(Vec<u8>, Vec<u8>)>>;

    struct Script {
        reply: Box<Reply>,
        sign: Box<Sign>,
        handshakes: Arc<Mutex<Vec<Option<HandshakeKind>>>>,
        requests: Arc<Mutex<Vec<Seen>>>,
        signatures: Arc<Mutex<Vec<Seen>>>,
        /// The last completion's request body and the response bytes sent.
        last: Last,
    }

    /// `sign(request body, response bytes)` answers the signature route.
    fn upstream_with(
        certs: impl Fn(usize) -> &'static str + Send + 'static,
        close_each: bool,
        reply: impl Fn(&Seen) -> Answer + Send + Sync + 'static,
        sign: impl Fn(&[u8], &[u8]) -> (u16, Vec<u8>) + Send + Sync + 'static,
    ) -> Upstream {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let handshakes = Arc::new(Mutex::new(Vec::new()));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let signatures = Arc::new(Mutex::new(Vec::new()));
        let script = Arc::new(Script {
            reply: Box::new(reply),
            sign: Box::new(sign),
            handshakes: handshakes.clone(),
            requests: requests.clone(),
            signatures: signatures.clone(),
            last: Mutex::new(None),
        });
        // One ServerConfig per certificate, shared across connections, so a
        // client that offered resumption would get it.
        std::thread::spawn(move || {
            let mut configs: Vec<(&'static str, Arc<rustls::ServerConfig>)> = Vec::new();
            for (i, tcp) in listener.incoming().enumerate() {
                let Ok(tcp) = tcp else { continue };
                let name = certs(i);
                let config = match configs.iter().find(|(n, _)| *n == name) {
                    Some((_, cfg)) => cfg.clone(),
                    None => {
                        let cfg = server_config(name);
                        configs.push((name, cfg.clone()));
                        cfg
                    }
                };
                let script = script.clone();
                std::thread::spawn(move || serve(tcp, config, close_each, &script));
            }
        });
        Upstream {
            port,
            handshakes,
            requests,
            signatures,
        }
    }

    fn serve(
        mut tcp: std::net::TcpStream,
        config: Arc<rustls::ServerConfig>,
        close_each: bool,
        script: &Script,
    ) {
        let handshakes = &script.handshakes;
        tcp.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        let mut conn = rustls::ServerConnection::new(config).unwrap();
        while conn.is_handshaking() {
            if conn.complete_io(&mut tcp).is_err() {
                handshakes.lock().unwrap().push(None);
                return;
            }
        }
        handshakes.lock().unwrap().push(conn.handshake_kind());
        let mut tls = rustls::StreamOwned::new(conn, tcp);
        loop {
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                match tls.read(&mut byte) {
                    Ok(1) => head.push(byte[0]),
                    _ => return,
                }
            }
            let head = String::from_utf8(head).unwrap();
            let length: usize = head
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse().unwrap())
                })
                .unwrap_or(0);
            let mut body = vec![0; length];
            if tls.read_exact(&mut body).is_err() {
                return;
            }
            let seen = Seen { head, body };
            let answer = if seen.head.starts_with("GET /v1/signature/") {
                script.signatures.lock().unwrap().push(seen.clone());
                let last = script.last.lock().unwrap().clone();
                let (status, out) = match last {
                    Some((req, resp)) => (script.sign)(&req, &resp),
                    None => (404, b"{}".to_vec()),
                };
                Answer::Whole(status, out)
            } else {
                script.requests.lock().unwrap().push(seen.clone());
                let answer = (script.reply)(&seen);
                *script.last.lock().unwrap() = Some((seen.body.clone(), answer.bytes()));
                answer
            };
            let close = if close_each {
                "Connection: close\r\n"
            } else {
                ""
            };
            match answer {
                Answer::Whole(status, out) => {
                    let _ = write!(
                        tls,
                        "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{close}\r\n",
                        out.len()
                    );
                    let _ = tls.write_all(&out);
                }
                Answer::Encoded(encoding, out) => {
                    let _ = write!(
                        tls,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Encoding: {encoding}\r\nContent-Length: {}\r\n{close}\r\n",
                        out.len()
                    );
                    let _ = tls.write_all(&out);
                }
                Answer::Sse {
                    chunks,
                    pauses,
                    last_sent,
                } => {
                    let _ = write!(
                        tls,
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n{close}\r\n"
                    );
                    let _ = tls.flush();
                    for (i, chunk) in chunks.iter().enumerate() {
                        std::thread::sleep(pauses.get(i).copied().unwrap_or_default());
                        if i + 1 == chunks.len() {
                            *last_sent.lock().unwrap() = Some(Instant::now());
                        }
                        let _ = write!(tls, "{:x}\r\n", chunk.len());
                        let _ = tls.write_all(chunk);
                        let _ = tls.write_all(b"\r\n");
                        let _ = tls.flush();
                    }
                    let _ = tls.write_all(b"0\r\n\r\n");
                }
            }
            let _ = tls.flush();
            if close_each {
                tls.conn.send_close_notify();
                let _ = tls.flush();
                return;
            }
        }
    }

    fn ok_reply(_: &Seen) -> Answer {
        Answer::Whole(200, REPLY.to_vec())
    }

    /// Counts calls and returns `answer`. Starting a runtime panics on a
    /// Tokio worker thread, exactly as `BlockingHttp` does, so this also
    /// proves `reattest` runs on the blocking pool.
    fn reattest(answer: &[&str], calls: &Arc<AtomicUsize>) -> Reattest {
        reattest_keys(answer, &[], calls)
    }

    /// As `reattest`, also attesting `keys` for model "m".
    fn reattest_keys(answer: &[&str], keys: &[[u8; 32]], calls: &Arc<AtomicUsize>) -> Reattest {
        let answer = fps(answer);
        let keys: BTreeSet<[u8; 32]> = keys.iter().copied().collect();
        let calls = calls.clone();
        Arc::new(move || {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap()
                .block_on(async {});
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Attested {
                fingerprints: answer.clone(),
                keys: if keys.is_empty() {
                    Default::default()
                } else {
                    [("m".to_string(), keys.clone())].into()
                },
                ..Attested::default()
            })
        })
    }

    /// As `reattest`, also attesting `gateway_keys`.
    fn reattest_gateway(
        answer: &[&str],
        gateway_keys: &[[u8; 32]],
        calls: &Arc<AtomicUsize>,
    ) -> Reattest {
        let answer = fps(answer);
        let gateway_keys: BTreeSet<[u8; 32]> = gateway_keys.iter().copied().collect();
        let calls = calls.clone();
        Arc::new(move || {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Attested {
                fingerprints: answer.clone(),
                gateway_keys: gateway_keys.clone(),
                ..Attested::default()
            })
        })
    }

    struct Rig {
        /// None once `stop_forwarder` has run.
        forwarder: Option<Forwarder>,
        fatal: Arc<Fatal>,
        record: Arc<AttestationRecord>,
        dir: std::path::PathBuf,
    }

    impl Rig {
        fn doc(&self) -> serde_json::Value {
            serde_json::from_slice(&std::fs::read(self.record.path()).unwrap()).unwrap()
        }
        fn post(&self, body: &[u8]) -> (u16, Vec<u8>) {
            send(self.fw(), "POST", "/v1/chat/completions", body)
        }
        fn fw(&self) -> &Forwarder {
            self.forwarder.as_ref().expect("forwarder running")
        }
        /// Drops the forwarder (its drain included); the files stay.
        fn stop_forwarder(&mut self) {
            drop(self.forwarder.take());
        }
        fn log_path(&self) -> std::path::PathBuf {
            self.record.session_dir().join("inference.jsonl")
        }
        fn lines(&self) -> Vec<serde_json::Value> {
            std::fs::read_to_string(self.log_path())
                .unwrap_or_default()
                .lines()
                .map(|l| serde_json::from_str(l).unwrap())
                .collect()
        }
        fn wait_lines(&self, n: usize) -> Vec<serde_json::Value> {
            wait_for("inference.jsonl lines", || self.lines().len() >= n);
            self.lines()
        }
    }

    impl Drop for Rig {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn rig(up: &Upstream, seed: &[&str], reattest: Reattest, max: usize) -> Rig {
        rig_with(up, seed, &[test_vk()], reattest, max, quick())
    }

    /// `keys` are the seed's attested keys for model "m".
    fn rig_with(
        up: &Upstream,
        seed: &[&str],
        keys: &[[u8; 32]],
        reattest: Reattest,
        max: usize,
        signatures: SignaturePolicy,
    ) -> Rig {
        rig_seeded(up, seed, keys, &[], reattest, max, signatures)
    }

    /// As `rig_with`, also seeding the attested gateway keys.
    fn rig_seeded(
        up: &Upstream,
        seed: &[&str],
        keys: &[[u8; 32]],
        gateway_keys: &[[u8; 32]],
        reattest: Reattest,
        max: usize,
        signatures: SignaturePolicy,
    ) -> Rig {
        let dir = crate::testutil::tempdir("forwarder");
        let record =
            AttestationRecord::create(&dir, "0123456789ab", &AttestOptions::default()).unwrap();
        let seed = Attested {
            fingerprints: fps(seed),
            keys: [("m".to_string(), keys.iter().copied().collect())].into(),
            gateway_keys: gateway_keys.iter().copied().collect(),
            ..Attested::default()
        };
        record.verified(&seed).unwrap();
        let fatal = Fatal::new();
        let forwarder = Forwarder::start(ForwarderConfig {
            upstream: up.url(),
            models: vec!["m".into()],
            roots: roots(),
            seed,
            reattest,
            fatal: fatal.clone(),
            record: record.clone(),
            max_request_bytes: max,
            pin_tls: true,
            signatures,
            inference_log: record.session_dir().join("inference.jsonl"),
        })
        .unwrap();
        Rig {
            forwarder: Some(forwarder),
            fatal,
            record,
            dir,
        }
    }

    fn send(f: &Forwarder, method: &str, path: &str, body: &[u8]) -> (u16, Vec<u8>) {
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap();
        let url = format!("{}{}", f.base_url().trim_end_matches("/v1"), path);
        let resp = client
            .request(method.parse().unwrap(), url)
            .bearer_auth(DUMMY_KEY)
            .header("content-type", "application/json")
            .body(body.to_vec())
            .send()
            .unwrap();
        let status = resp.status().as_u16();
        (status, resp.bytes().unwrap().to_vec())
    }

    fn no_key_in(rig: &Rig) {
        let text = std::fs::read_to_string(rig.record.path()).unwrap();
        assert!(!text.contains(DUMMY_KEY));
        let log = std::fs::read_to_string(rig.log_path()).unwrap_or_default();
        assert!(!log.contains(DUMMY_KEY));
        assert!(!log.contains("Bearer"));
        assert!(!rig.fatal.reason().unwrap_or_default().contains(DUMMY_KEY));
        assert!(!format!("{:?}", rig.forwarder).contains(DUMMY_KEY));
    }

    #[test]
    fn a_pinned_upstream_gets_the_exact_bytes_and_the_key() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(
            rig.fw().base_url(),
            format!("http://127.0.0.1:{}/v1", rig.fw().port)
        );
        let (status, body) = rig.post(BODY);
        assert_eq!(status, 200);
        assert_eq!(body, REPLY);
        let seen = up.requests();
        assert_eq!(seen.len(), 1);
        let head = seen[0].head.to_ascii_lowercase();
        assert!(
            seen[0]
                .head
                .starts_with("POST /v1/chat/completions HTTP/1.1\r\n")
        );
        assert!(
            head.lines()
                .any(|l| l == format!("authorization: bearer {}", DUMMY_KEY.to_ascii_lowercase())),
            "{head}"
        );
        assert!(
            head.lines().any(|l| l == "accept-encoding: identity"),
            "{head}"
        );
        assert!(
            head.lines().any(|l| l == "content-type: application/json"),
            "{head}"
        );
        assert_eq!(seen[0].body, BODY);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert_eq!(rig.fatal.reason(), None);
        assert_eq!(rig.doc()["status"], "verified");
        no_key_in(&rig);
    }

    #[test]
    fn only_the_three_inbound_headers_are_forwarded() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .build()
            .unwrap();
        let resp = client
            .post(format!("{}/chat/completions", rig.fw().base_url()))
            .bearer_auth(DUMMY_KEY)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .header("x-ironwire-route", "nearai:m")
            .header("cookie", "c=1")
            .header("accept-encoding", "gzip")
            .body(BODY.to_vec())
            .send()
            .unwrap();
        assert_eq!(resp.status(), 200);
        let head = up.requests()[0].head.to_ascii_lowercase();
        let names: BTreeSet<&str> = head
            .lines()
            .skip(1)
            .filter_map(|l| l.split_once(':').map(|(n, _)| n))
            .collect();
        assert_eq!(
            names,
            BTreeSet::from([
                "accept",
                "accept-encoding",
                "authorization",
                "content-length",
                "content-type",
                "host"
            ]),
            "{head}"
        );
        assert!(head.lines().any(|l| l == "accept: text/event-stream"));
        assert!(head.lines().any(|l| l == "accept-encoding: identity"));
    }

    #[test]
    fn an_unpinned_certificate_never_sees_a_request_and_ends_the_session() {
        let up = upstream(|_| "b", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let (status, _) = rig.post(BODY);
        assert_eq!(status, 502);
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(up.requests().is_empty());
        let reason = rig.fatal.reason().expect("fatal");
        assert!(reason.contains("pin"), "{reason}");
        assert!(reason.contains(&hex::encode(fp("b"))), "{reason}");
        let doc = rig.doc();
        assert_eq!(doc["status"], "pin-failed");
        assert_eq!(doc["reattestations"].as_array().unwrap().len(), 3);
        wait_for("the rejected handshake", || up.handshakes().len() == 1);
        assert_eq!(up.handshakes(), vec![None]);
        let (status, _) = rig.post(BODY);
        assert_eq!(status, 502);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(up.handshakes().len(), 1, "no new handshake after fatal");
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "no new reattest after fatal"
        );
        no_key_in(&rig);
    }

    #[test]
    fn a_rotated_certificate_is_accepted_after_one_reattestation() {
        let up = upstream(|_| "b", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["b"], &calls), 1 << 20);
        let (status, body) = rig.post(BODY);
        assert_eq!(status, 200);
        assert_eq!(body, REPLY);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(up.requests().len(), 1);
        assert_eq!(up.requests()[0].body, BODY);
        let doc = rig.doc();
        assert_eq!(doc["status"], "verified");
        let re = doc["reattestations"].as_array().unwrap();
        assert_eq!(re.len(), 1);
        assert_eq!(re[0]["fingerprints"][0], hex::encode(fp("b")));
        assert_eq!(rig.fatal.reason(), None);
    }

    /// Review finding: a certificate rotated between a completion and its
    /// receipt fetch is first seen by the fetch. It is learned like one seen
    /// by a completion (re-attestation), and the fetch is retried.
    #[test]
    fn a_certificate_rotated_before_the_receipt_fetch_is_learned() {
        let up = upstream(|i| if i == 0 { "a" } else { "b" }, true, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a", "b"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY), (200, REPLY.to_vec()));
        let lines = rig.wait_lines(1);
        assert_eq!(lines[0]["result"], "verified", "{lines:?}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(up.signature_requests().len(), 1);
        let doc = rig.doc();
        assert_eq!(doc["status"], "verified");
        assert_eq!(doc["reattestations"].as_array().unwrap().len(), 1);
        assert_eq!(rig.fatal.reason(), None);
        no_key_in(&rig);
    }

    /// As above, but re-attestation never attests the new certificate: the
    /// session ends `pin-failed`, not `signature-failed`.
    #[test]
    fn a_rotated_certificate_the_receipt_fetch_cannot_learn_is_a_pin_failure() {
        let up = upstream(|i| if i == 0 { "a" } else { "b" }, true, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY), (200, REPLY.to_vec()));
        let lines = rig.wait_lines(1);
        let result = lines[0]["result"].as_str().unwrap();
        assert!(result.starts_with("failed: pin"), "{result}");
        // `stop` raises fatal, then writes the record.
        wait_for("pin-failed", || rig.doc()["status"] == "pin-failed");
        let reason = rig.fatal.reason().expect("fatal");
        assert!(reason.starts_with("pin-failed"), "{reason}");
        assert_eq!(rig.doc()["status"], "pin-failed");
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert!(up.signature_requests().is_empty());
        no_key_in(&rig);
    }

    #[test]
    fn a_pinned_certificate_from_an_untrusted_ca_is_refused() {
        let up = upstream(|_| "c", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["c"], reattest(&["c"], &calls), 1 << 20);
        let (status, _) = rig.post(BODY);
        assert_eq!(status, 502);
        assert!(up.requests().is_empty());
        let reason = rig.fatal.reason().expect("fatal");
        assert!(reason.contains("pin"), "{reason}");
        assert_eq!(rig.doc()["status"], "pin-failed");
        // Re-attestation cannot make a certificate chain to a trusted root.
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn every_upstream_handshake_is_full() {
        let up = upstream(|_| "a", true, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY).0, 200);
        assert_eq!(rig.post(BODY).0, 200);
        // Two completions and two signature fetches, each its own connection.
        let lines = rig.wait_lines(2);
        assert!(lines.iter().all(|l| l["result"] == "verified"), "{lines:?}");
        wait_for("four handshakes", || up.handshakes().len() == 4);
        assert!(
            up.handshakes()
                .iter()
                .all(|h| *h == Some(HandshakeKind::Full)),
            "{:?}",
            up.handshakes()
        );
    }

    /// Regression for the 2026-10-28 bun measurement: a pin checked per
    /// request can be skipped by a pooled connection. The pin lives in the
    /// TLS verifier, so every new connection is checked, including a second
    /// one made after the first was closed.
    #[test]
    fn a_second_connection_to_a_different_certificate_fails_the_pin() {
        let up = upstream(|i| if i == 0 { "a" } else { "b" }, true, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY).0, 200);
        assert_eq!(rig.post(BODY).0, 502);
        assert_eq!(
            up.requests().len(),
            1,
            "the unpinned connection saw a request"
        );
        assert!(rig.fatal.reason().unwrap().contains("pin"));
        assert_eq!(rig.doc()["status"], "pin-failed");
        wait_for("both connections", || up.handshakes().len() >= 2);
        assert_eq!(up.handshakes()[0], Some(HandshakeKind::Full));
        assert!(up.handshakes()[1..].iter().all(Option::is_none));
    }

    #[test]
    fn a_model_outside_the_allowlist_gets_403_and_no_connection() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let (status, _) = rig.post(br#"{"model":"other","messages":[]}"#);
        assert_eq!(status, 403);
        let reason = rig.fatal.reason().expect("fatal");
        assert!(reason.contains("model-refused"), "{reason}");
        assert_eq!(rig.doc()["status"], "model-refused");
        // A duplicate key is a parser differential: refused, never forwarded.
        let (status, _) = rig.post(br#"{"model":"m","model":"other"}"#);
        assert_eq!(status, 502, "fatal already raised");
        std::thread::sleep(Duration::from_millis(200));
        assert!(up.handshakes().is_empty());
        assert!(up.requests().is_empty());
    }

    #[test]
    fn only_chat_completions_is_forwarded() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        for (method, path) in [
            ("GET", "/v1/models"),
            ("POST", "/v1/responses"),
            ("GET", "/v1/signature/x"),
            ("GET", "/v1/chat/completions"),
            ("POST", "/v1/chat/completions/x"),
            ("POST", "/chat/completions"),
        ] {
            assert_eq!(send(rig.fw(), method, path, BODY).0, 404, "{method} {path}");
        }
        std::thread::sleep(Duration::from_millis(200));
        assert!(up.handshakes().is_empty());
        assert_eq!(rig.fatal.reason(), None);
    }

    #[test]
    fn malformed_or_oversized_requests_are_refused_locally() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let max = 4096;
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), max);
        for body in [
            &b"not json"[..],
            br#"{"messages":[]}"#,
            br#"{"model":7}"#,
            br#"["m"]"#,
            br#"{"model":"m","model":"m"}"#,
        ] {
            assert_eq!(rig.post(body).0, 400, "{}", String::from_utf8_lossy(body));
        }
        let mut big = br#"{"model":"m","pad":""#.to_vec();
        big.resize(max - 2, b'x');
        big.extend_from_slice(br#""}"#);
        assert_eq!(big.len(), max);
        assert_eq!(
            rig.post(&big).0,
            200,
            "exactly max_request_bytes is allowed"
        );
        big.insert(big.len() - 2, b'x');
        assert_eq!(rig.post(&big).0, 413);
        assert_eq!(up.requests().len(), 1);
        assert_eq!(rig.fatal.reason(), None);
    }

    #[test]
    fn ironwire_reaches_near_ai_only_through_the_forwarder() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let inference = crate::inference::start_with_endpoint(
            DUMMY_KEY.to_string().into(),
            &["m".into()],
            &rig.fw().base_url(),
        )
        .unwrap();
        let client = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap();
        let resp = client
            .post(format!("{}/chat/completions", inference.provider().base_url))
            .bearer_auth(crate::config::PLACEHOLDER_API_KEY)
            .header("x-ironwire-route", crate::inference::route_header("m").unwrap())
            .json(&serde_json::json!({"model":"m","messages":[{"role":"user","content":"hi"}],"stream":false}))
            .send()
            .unwrap();
        assert_eq!(resp.status(), 200);
        let seen = up.requests();
        assert_eq!(seen.len(), 1);
        assert!(
            seen[0]
                .head
                .starts_with("POST /v1/chat/completions HTTP/1.1\r\n")
        );
        assert!(
            seen[0]
                .head
                .lines()
                .any(|l| l.eq_ignore_ascii_case(&format!("authorization: Bearer {DUMMY_KEY}")))
        );
        let body: serde_json::Value = serde_json::from_slice(&seen[0].body).unwrap();
        assert_eq!(body["model"], "m");
    }

    #[test]
    fn start_refuses_an_empty_seed_and_a_plain_http_upstream() {
        let dir = crate::testutil::tempdir("forwarder-start");
        let record =
            AttestationRecord::create(&dir, "0123456789ab", &AttestOptions::default()).unwrap();
        let calls = Arc::new(AtomicUsize::new(0));
        let cfg = |upstream: &str, seed: &[&str]| ForwarderConfig {
            upstream: upstream.into(),
            models: vec!["m".into()],
            roots: roots(),
            seed: Attested {
                fingerprints: fps(seed),
                ..Attested::default()
            },
            reattest: reattest(&["a"], &calls),
            fatal: Fatal::new(),
            record: record.clone(),
            max_request_bytes: 1024,
            pin_tls: true,
            signatures: SignaturePolicy::default(),
            inference_log: record.session_dir().join("inference.jsonl"),
        };
        assert!(Forwarder::start(cfg("https://127.0.0.1:1/v1", &[])).is_err());
        assert!(Forwarder::start(cfg("http://127.0.0.1:1/v1", &["a"])).is_err());
        assert!(Forwarder::start(cfg("https://127.0.0.1:1/v1", &["a"])).is_ok());

        // Signature mode (no TLS pin): fingerprints are not required, but the
        // attested model signing keys are, since they are the only binding.
        let sig = |keyed: bool| {
            let mut seed = Attested::default();
            if keyed {
                seed.keys.insert("m".into(), std::collections::BTreeSet::from([[7u8; 32]]));
            }
            ForwarderConfig {
                upstream: "https://127.0.0.1:1/v1".into(),
                models: vec!["m".into()],
                roots: roots(),
                seed,
                reattest: reattest(&["a"], &calls),
                fatal: Fatal::new(),
                record: record.clone(),
                max_request_bytes: 1024,
                pin_tls: false,
                signatures: SignaturePolicy::default(),
                inference_log: record.session_dir().join("inference.jsonl"),
            }
        };
        assert!(Forwarder::start(sig(false)).is_err(), "no keys in signature mode");
        assert!(Forwarder::start(sig(true)).is_ok(), "no pin needed in signature mode");
        std::fs::remove_dir_all(dir).unwrap();
    }

    fn post_to(base_url: &str, body: &[u8]) -> u16 {
        reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap()
            .post(format!("{base_url}/chat/completions"))
            .bearer_auth(DUMMY_KEY)
            .body(body.to_vec())
            .send()
            .unwrap()
            .status()
            .as_u16()
    }

    /// Review finding: a request parked on the re-attestation lock while
    /// another request's attempts end in pin-failed must not re-attest again
    /// or reach upstream.
    #[test]
    fn parked_requests_share_one_round_of_reattestation_and_stop_after_fatal() {
        let up = upstream(|_| "b", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let (first_tx, first_rx) = std::sync::mpsc::channel();
        let first_tx = Mutex::new(first_tx);
        let counted = calls.clone();
        let seen = up.handshakes.clone();
        let slow: Reattest = Arc::new(move || {
            if counted.fetch_add(1, Ordering::SeqCst) == 0 {
                let _ = first_tx.lock().unwrap().send(());
                // A's first re-attestation holds the lock until B's own
                // handshake has been refused, so B is always parked behind
                // A rather than arriving after the fatal (no fixed sleep).
                let deadline = Instant::now() + Duration::from_secs(10);
                while seen.lock().unwrap().len() < 2 && Instant::now() < deadline {
                    std::thread::sleep(Duration::from_millis(5));
                }
            }
            Ok(Attested {
                fingerprints: fps(&["a"]),
                ..Attested::default()
            })
        });
        let rig = rig(&up, &["a"], slow, 1 << 20);
        let base = rig.fw().base_url();
        let a = {
            let base = base.clone();
            std::thread::spawn(move || post_to(&base, BODY))
        };
        // B starts while A is inside its first re-attestation, which waits
        // for B's refused handshake, so B takes its own Unpinned rejection
        // and parks on the lock.
        first_rx.recv_timeout(Duration::from_secs(10)).unwrap();
        let b = std::thread::spawn(move || post_to(&base, BODY));
        assert_eq!(a.join().unwrap(), 502);
        assert_eq!(b.join().unwrap(), 502);
        wait_for("both rejected handshakes", || up.handshakes().len() == 2);
        assert_eq!(calls.load(Ordering::SeqCst), 3, "B re-attested after fatal");
        assert!(up.requests().is_empty());
        let doc = rig.doc();
        assert_eq!(doc["status"], "pin-failed");
        assert_eq!(doc["reattestations"].as_array().unwrap().len(), 3);
        assert!(rig.fatal.reason().unwrap().starts_with("pin-failed"));
    }

    /// Review finding: fatal raised while a request body is still arriving
    /// must stop that request before it is sent.
    #[test]
    fn a_fatal_raised_while_the_body_arrives_stops_the_request() {
        use std::net::TcpStream;
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let mut s = TcpStream::connect(("127.0.0.1", rig.fw().port)).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write!(
            s,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer {DUMMY_KEY}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            BODY.len()
        )
        .unwrap();
        s.write_all(&BODY[..10]).unwrap();
        s.flush().unwrap();
        std::thread::sleep(Duration::from_millis(200));
        rig.fatal.raise("signature-failed: x");
        s.write_all(&BODY[10..]).unwrap();
        let mut resp = String::new();
        s.read_to_string(&mut resp).unwrap();
        assert!(resp.starts_with("HTTP/1.1 502"), "{resp}");
        std::thread::sleep(Duration::from_millis(200));
        assert!(up.handshakes().is_empty());
        assert!(up.requests().is_empty());
    }

    #[test]
    fn the_first_fatal_owns_the_record_status() {
        let dir = crate::testutil::tempdir("forwarder-stop");
        let record =
            AttestationRecord::create(&dir, "0123456789ab", &AttestOptions::default()).unwrap();
        let fatal = Fatal::new();
        assert!(stop(&fatal, &record, "model-refused", "model \"x\""));
        assert!(!stop(&fatal, &record, "pin-failed", "pin: later"));
        let doc: serde_json::Value =
            serde_json::from_slice(&std::fs::read(record.path()).unwrap()).unwrap();
        assert_eq!(doc["status"], "model-refused");
        assert_eq!(doc["reason"], "model \"x\"");
        assert_eq!(fatal.reason().unwrap(), "model-refused: model \"x\"");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn a_refused_path_is_logged_by_method_and_path_only() {
        assert_eq!(
            not_forwarded_line(&Method::GET, "/v1/models"),
            "[rts forwarder] 404 GET /v1/models: not forwarded"
        );
    }

    #[test]
    fn fatal_keeps_its_first_reason() {
        let f = Fatal::new();
        assert_eq!(f.reason(), None);
        assert!(f.raise("pin-failed: one"));
        assert!(!f.raise("model-refused: two"));
        assert_eq!(f.reason().as_deref(), Some("pin-failed: one"));
    }

    fn wait_within(limit: Duration, what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + limit;
        while !cond() {
            assert!(Instant::now() < deadline, "not within {limit:?}: {what}");
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn stream_client() -> reqwest::blocking::Client {
        reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap()
    }

    fn post_stream(rig: &Rig) -> reqwest::blocking::Response {
        stream_client()
            .post(format!("{}/chat/completions", rig.fw().base_url()))
            .bearer_auth(DUMMY_KEY)
            .header("content-type", "application/json")
            .header("accept", "text/event-stream")
            .body(STREAM_BODY.to_vec())
            .send()
            .unwrap()
    }

    #[test]
    fn a_verified_completion_is_logged_and_the_session_continues() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let (status, body) = rig.post(BODY);
        assert_eq!(status, 200);
        assert_eq!(body, REPLY);
        let lines = rig.wait_lines(1);
        assert_eq!(lines.len(), 1, "{lines:?}");
        let l = &lines[0];
        assert_eq!(l["result"], "verified", "{l}");
        assert_eq!(l["chat_id"], "x");
        assert_eq!(l["model"], "m");
        assert_eq!(l["streamed"], false);
        assert_eq!(l["request_sha256"], sha_hex(BODY));
        assert_eq!(l["response_sha256"], sha_hex(REPLY));
        assert_eq!(l["signing_key"], hex::encode(test_vk()));
        assert!(l["signature_ms"].is_u64(), "{l}");
        assert!(l["ts"].as_u64().unwrap() > 1_700_000_000, "{l}");
        let sig = up.signature_requests();
        assert_eq!(sig.len(), 1);
        assert!(
            sig[0]
                .head
                .starts_with("GET /v1/signature/x?model=m&signing_algo=ed25519 HTTP/1.1\r\n"),
            "{}",
            sig[0].head
        );
        assert!(
            sig[0]
                .head
                .lines()
                .any(|l| l.eq_ignore_ascii_case(&format!("authorization: Bearer {DUMMY_KEY}")))
        );
        assert_eq!(rig.fatal.reason(), None);
        assert_eq!(rig.doc()["status"], "verified");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        // The session continues.
        assert_eq!(rig.post(BODY).0, 200);
        let lines = rig.wait_lines(2);
        assert_eq!(lines[1]["result"], "verified");
        assert_eq!(rig.fatal.reason(), None);
        no_key_in(&rig);
    }

    #[test]
    fn a_verified_stream_is_logged_and_was_not_buffered() {
        let last_sent = Arc::new(Mutex::new(None));
        let sent = last_sent.clone();
        let pauses = vec![Duration::ZERO, Duration::ZERO, Duration::from_millis(200)];
        let up = upstream(|_| "a", false, move |_| sse_answer(pauses.clone(), &sent));
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let mut resp = post_stream(&rig);
        assert_eq!(resp.status(), 200);
        assert_eq!(
            resp.headers().get("content-type").unwrap(),
            "text/event-stream"
        );
        let mut buf = [0u8; 4096];
        let n = resp.read(&mut buf).unwrap();
        assert!(n > 0);
        let first_at = Instant::now();
        let mut all = buf[..n].to_vec();
        resp.read_to_end(&mut all).unwrap();
        assert_eq!(all, SSE.concat());
        let sent_at = last_sent.lock().unwrap().expect("last chunk sent");
        assert!(
            first_at < sent_at,
            "the client got its first bytes only after upstream sent the last chunk"
        );
        let lines = rig.wait_lines(1);
        let l = &lines[0];
        assert_eq!(l["result"], "verified", "{l}");
        assert_eq!(l["streamed"], true);
        assert_eq!(l["chat_id"], "chatcmpl-s1");
        assert_eq!(l["request_sha256"], sha_hex(STREAM_BODY));
        assert_eq!(l["response_sha256"], sha_hex(&SSE.concat()));
        assert!(
            up.signature_requests()[0]
                .head
                .starts_with("GET /v1/signature/chatcmpl-s1?model=m&signing_algo=ed25519 ")
        );
        assert_eq!(rig.fatal.reason(), None);
        no_key_in(&rig);
    }

    /// Carried from Task 5: a client that goes away mid-stream does not stop
    /// the relay reading upstream, so the hash covers the whole response.
    #[test]
    fn a_stream_the_client_dropped_is_read_to_the_end_and_verified() {
        let last_sent = Arc::new(Mutex::new(None));
        let sent = last_sent.clone();
        let up = upstream(
            |_| "a",
            false,
            move |_| Answer::Sse {
                chunks: (0..8)
                    .map(|i| format!("data: {{\"id\":\"chatcmpl-d\",\"n\":{i}}}\n\n").into_bytes())
                    .collect(),
                pauses: vec![Duration::from_millis(100); 8],
                last_sent: sent.clone(),
            },
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let mut resp = post_stream(&rig);
        let mut buf = [0u8; 64];
        assert!(resp.read(&mut buf).unwrap() > 0);
        drop(resp);
        let lines = rig.wait_lines(1);
        let l = &lines[0];
        assert_eq!(l["result"], "verified", "{l}");
        assert_eq!(l["chat_id"], "chatcmpl-d");
        assert!(last_sent.lock().unwrap().is_some());
        assert_eq!(rig.fatal.reason(), None);
    }

    /// Review finding: the receipt covers the bytes as NEAR AI sent them, so
    /// an encoded body (which the forwarder never asked for) fails closed,
    /// even with a receipt that matches; `identity` is no encoding.
    #[test]
    fn an_encoded_response_ends_the_session() {
        let up = upstream(|_| "a", false, |_| Answer::Encoded("gzip", REPLY.to_vec()));
        let calls = Arc::new(AtomicUsize::new(0));
        let encoded = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let resp = reqwest::blocking::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(20))
            .build()
            .unwrap()
            .post(format!("{}/chat/completions", encoded.fw().base_url()))
            .bearer_auth(DUMMY_KEY)
            .body(BODY.to_vec())
            .send()
            .unwrap();
        assert_eq!(resp.status(), 200);
        // Nothing is relayed: the client's body ends in an error.
        assert!(resp.bytes().is_err(), "an encoded body reached the client");
        let lines = encoded.wait_lines(1);
        let result = lines[0]["result"].as_str().unwrap();
        assert!(
            result.starts_with("failed: ") && result.contains("Content-Encoding"),
            "{result}"
        );
        // `stop` raises fatal, then writes the record.
        wait_for("signature-failed", || {
            encoded.doc()["status"] == "signature-failed"
        });
        assert!(
            encoded
                .fatal
                .reason()
                .unwrap()
                .starts_with("signature-failed")
        );
        assert_eq!(encoded.doc()["status"], "signature-failed");
        no_key_in(&encoded);

        let up = upstream(
            |_| "a",
            false,
            |_| Answer::Encoded("identity", REPLY.to_vec()),
        );
        let plain = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(plain.post(BODY), (200, REPLY.to_vec()));
        assert_eq!(plain.wait_lines(1)[0]["result"], "verified");
        assert_eq!(plain.fatal.reason(), None);
    }

    #[test]
    fn a_wrong_receipt_ends_the_session() {
        let up = upstream_with(|_| "a", false, ok_reply, wrong_receipt);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let (status, body) = rig.post(BODY);
        assert_eq!(status, 200, "relayed before the check");
        assert_eq!(body, REPLY);
        wait_within(Duration::from_secs(2), "fatal", || {
            rig.fatal.reason().is_some()
        });
        let reason = rig.fatal.reason().unwrap();
        assert!(reason.contains("signature"), "{reason}");
        wait_for("signature-failed", || {
            rig.doc()["status"] == "signature-failed"
        });
        let doc = rig.doc();
        assert!(
            doc["reason"].as_str().unwrap().contains("response"),
            "{doc}"
        );
        let lines = rig.wait_lines(1);
        let result = lines[0]["result"].as_str().unwrap();
        assert!(result.starts_with("failed:"), "{result}");
        assert_eq!(lines[0]["signing_key"], serde_json::Value::Null);
        let handshakes = up.handshakes().len();
        assert_eq!(rig.post(BODY).0, 502);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(
            up.handshakes().len(),
            handshakes,
            "no handshake after fatal"
        );
        assert_eq!(up.requests().len(), 1);
        assert_eq!(up.signature_requests().len(), 1);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a known key never re-attests"
        );
        no_key_in(&rig);
    }

    #[test]
    fn a_receipt_that_never_arrives_ends_the_session() {
        let up = upstream_with(|_| "a", false, ok_reply, |_, _| (404, b"{}".to_vec()));
        let calls = Arc::new(AtomicUsize::new(0));
        let policy = SignaturePolicy {
            window: Duration::from_millis(300),
            ..quick()
        };
        let rig = rig_with(
            &up,
            &["a"],
            &[test_vk()],
            reattest(&["a"], &calls),
            1 << 20,
            policy,
        );
        assert_eq!(rig.post(BODY).0, 200);
        wait_within(Duration::from_secs(3), "fatal", || {
            rig.fatal.reason().is_some()
        });
        assert!(rig.fatal.reason().unwrap().contains("signature"));
        // stop() raises fatal first, then writes the status.
        wait_for("signature-failed", || {
            rig.doc()["status"] == "signature-failed"
        });
        let lines = rig.wait_lines(1);
        let result = lines[0]["result"].as_str().unwrap();
        assert!(result.starts_with("failed:"), "{result}");
        assert!(result.contains("no signature within"), "{result}");
        assert!(up.signature_requests().len() >= 2);
        no_key_in(&rig);
    }

    #[test]
    fn an_unattested_key_is_reattested_once() {
        // Re-attestation attests the key: verified.
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig_with(
            &up,
            &["a"],
            &[],
            reattest_keys(&["a"], &[test_vk()], &calls),
            1 << 20,
            quick(),
        );
        assert_eq!(rig.post(BODY).0, 200);
        let lines = rig.wait_lines(1);
        assert_eq!(lines[0]["result"], "verified", "{}", lines[0]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let doc = rig.doc();
        assert_eq!(doc["reattestations"].as_array().unwrap().len(), 1);
        assert_eq!(doc["status"], "verified");
        assert_eq!(rig.fatal.reason(), None);
        // The merged key is kept: the next receipt needs no re-attestation.
        assert_eq!(rig.post(BODY).0, 200);
        assert_eq!(rig.wait_lines(2)[1]["result"], "verified");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Re-attestation does not attest it: fatal after exactly one.
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig_with(&up, &["a"], &[], reattest(&["a"], &calls), 1 << 20, quick());
        assert_eq!(rig.post(BODY).0, 200);
        wait_within(Duration::from_secs(3), "fatal", || {
            rig.fatal.reason().is_some()
        });
        let reason = rig.fatal.reason().unwrap();
        assert!(reason.contains("signature"), "{reason}");
        assert!(reason.contains("not attested"), "{reason}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        // stop() raises fatal first, then writes the status.
        wait_for("signature-failed", || {
            rig.doc()["status"] == "signature-failed"
        });
        assert!(
            rig.wait_lines(1)[0]["result"]
                .as_str()
                .unwrap()
                .starts_with("failed:")
        );
    }

    /// NEAR AI production signs receipts with the attested gateway's key
    /// (session d493b14cec32, 2026-10-29): accepted for any allowed model.
    #[test]
    fn a_receipt_signed_by_the_attested_gateway_key_verifies() {
        let up = upstream_with(|_| "a", false, ok_reply, gateway_receipt);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig_seeded(
            &up,
            &["a"],
            &[test_vk()],
            &[gateway_vk()],
            reattest(&["a"], &calls),
            1 << 20,
            quick(),
        );
        assert_eq!(rig.post(BODY), (200, REPLY.to_vec()));
        let lines = rig.wait_lines(1);
        assert_eq!(lines[0]["result"], "verified", "{}", lines[0]);
        assert_eq!(lines[0]["signer"], "gateway");
        assert_eq!(lines[0]["signing_key"], hex::encode(gateway_vk()));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "a known key never re-attests"
        );
        assert_eq!(rig.fatal.reason(), None);
        assert_eq!(
            rig.doc()["gateway_keys"],
            serde_json::json!([hex::encode(gateway_vk())])
        );
        no_key_in(&rig);
    }

    #[test]
    fn a_receipt_signed_by_a_model_key_verifies_and_logs_signer_model() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig_seeded(
            &up,
            &["a"],
            &[test_vk()],
            &[gateway_vk()],
            reattest(&["a"], &calls),
            1 << 20,
            quick(),
        );
        assert_eq!(rig.post(BODY).0, 200);
        let lines = rig.wait_lines(1);
        assert_eq!(lines[0]["result"], "verified", "{}", lines[0]);
        assert_eq!(lines[0]["signer"], "model");
        assert_eq!(lines[0]["signing_key"], hex::encode(test_vk()));
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    /// A key that is neither the model's nor the gateway's gets exactly one
    /// re-attestation; one that does not attest it ends the session.
    #[test]
    fn a_key_neither_model_nor_gateway_fails_after_one_reattestation() {
        let up = upstream_with(|_| "a", false, ok_reply, gateway_receipt);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig_seeded(
            &up,
            &["a"],
            &[test_vk()],
            &[[0x77; 32]],
            reattest_gateway(&["a"], &[[0x78; 32]], &calls),
            1 << 20,
            quick(),
        );
        assert_eq!(rig.post(BODY).0, 200);
        wait_within(Duration::from_secs(3), "fatal", || {
            rig.fatal.reason().is_some()
        });
        let reason = rig.fatal.reason().unwrap();
        assert!(reason.contains("not attested"), "{reason}");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        wait_for("signature-failed", || {
            rig.doc()["status"] == "signature-failed"
        });
        let lines = rig.wait_lines(1);
        assert!(lines[0]["result"].as_str().unwrap().starts_with("failed:"));
        assert_eq!(lines[0]["signer"], serde_json::Value::Null);
        assert_eq!(lines[0]["signing_key"], serde_json::Value::Null);
        no_key_in(&rig);
    }

    /// A re-attestation that fails a check (here, the gateway's) yields no
    /// gateway key: the receipt it signed is refused.
    #[test]
    fn a_gateway_key_from_a_failed_reattestation_is_not_accepted() {
        let up = upstream_with(|_| "a", false, ok_reply, gateway_receipt);
        let calls = Arc::new(AtomicUsize::new(0));
        let failing: Reattest = {
            let calls = calls.clone();
            Arc::new(move || {
                calls.fetch_add(1, Ordering::SeqCst);
                Err("model m: gateway: TDX quote: bad signature".into())
            })
        };
        let rig = rig_seeded(&up, &["a"], &[test_vk()], &[], failing, 1 << 20, quick());
        assert_eq!(rig.post(BODY).0, 200);
        wait_for("signature-failed", || {
            rig.doc()["status"] == "signature-failed"
        });
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let lines = rig.wait_lines(1);
        assert!(lines[0]["result"].as_str().unwrap().starts_with("failed:"));
        assert_eq!(lines[0]["signer"], serde_json::Value::Null);
    }

    /// Re-attestation carries gateway keys into the forwarder and the record.
    #[test]
    fn a_gateway_key_learned_by_reattestation_is_kept() {
        let up = upstream_with(|_| "a", false, ok_reply, gateway_receipt);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig_seeded(
            &up,
            &["a"],
            &[test_vk()],
            &[],
            reattest_gateway(&["a"], &[gateway_vk()], &calls),
            1 << 20,
            quick(),
        );
        assert_eq!(rig.post(BODY).0, 200);
        let lines = rig.wait_lines(1);
        assert_eq!(lines[0]["result"], "verified", "{}", lines[0]);
        assert_eq!(lines[0]["signer"], "gateway");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let doc = rig.doc();
        assert_eq!(
            doc["reattestations"][0]["gateway_keys"],
            serde_json::json!([hex::encode(gateway_vk())])
        );
        assert_eq!(rig.post(BODY).0, 200);
        assert_eq!(rig.wait_lines(2)[1]["signer"], "gateway");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the merged key is kept");
        assert_eq!(rig.fatal.reason(), None);
    }

    #[test]
    fn a_non_2xx_answer_is_relayed_and_logged_unsigned() {
        let up = upstream(
            |_| "a",
            false,
            |_| Answer::Whole(429, br#"{"error":"slow down"}"#.to_vec()),
        );
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let (status, body) = rig.post(BODY);
        assert_eq!(status, 429);
        assert_eq!(body, br#"{"error":"slow down"}"#);
        let lines = rig.wait_lines(1);
        let l = &lines[0];
        assert_eq!(l["result"], "not-signed (HTTP 429)", "{l}");
        assert_eq!(l["chat_id"], "");
        assert_eq!(l["request_sha256"], sha_hex(BODY));
        assert_eq!(l["signing_key"], serde_json::Value::Null);
        std::thread::sleep(Duration::from_millis(200));
        assert!(up.signature_requests().is_empty());
        assert_eq!(rig.fatal.reason(), None);
        assert_eq!(rig.doc()["status"], "verified");
        no_key_in(&rig);
    }

    /// Decision for Task 6: a fatal raised elsewhere cuts a response still
    /// streaming; nothing more reaches the model and no receipt is fetched.
    #[test]
    fn a_stream_in_flight_is_cut_when_fatal_is_raised() {
        let last_sent = Arc::new(Mutex::new(None));
        let sent = last_sent.clone();
        let pauses = vec![Duration::ZERO, Duration::ZERO, Duration::from_millis(1500)];
        let up = upstream(|_| "a", false, move |_| sse_answer(pauses.clone(), &sent));
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        let mut resp = post_stream(&rig);
        let mut buf = [0u8; 4096];
        let n = resp.read(&mut buf).unwrap();
        assert!(n > 0);
        let raised = Instant::now();
        assert!(rig.fatal.raise("model-refused: elsewhere"));
        let mut rest = buf[..n].to_vec();
        let _ = resp.read_to_end(&mut rest);
        assert!(
            raised.elapsed() < Duration::from_millis(1000),
            "the stream was not cut"
        );
        assert!(!rest.ends_with(SSE[2]), "the last chunk reached the client");
        let lines = rig.wait_lines(1);
        assert_eq!(
            lines[0]["result"], "failed: session stopped",
            "{}",
            lines[0]
        );
        std::thread::sleep(Duration::from_millis(200));
        assert!(up.signature_requests().is_empty());
        assert_eq!(rig.fatal.reason().unwrap(), "model-refused: elsewhere");
    }

    #[test]
    fn the_logs_never_hold_the_key() {
        // Verified, then unsigned (429), then a wrong receipt.
        let n = Arc::new(AtomicUsize::new(0));
        let count = n.clone();
        let reply = move |_: &Seen| match count.fetch_add(1, Ordering::SeqCst) {
            1 => Answer::Whole(429, br#"{"error":"slow"}"#.to_vec()),
            _ => Answer::Whole(200, REPLY.to_vec()),
        };
        let signed = Arc::new(AtomicUsize::new(0));
        let sign = move |req: &[u8], resp: &[u8]| {
            if signed.fetch_add(1, Ordering::SeqCst) == 0 {
                good_receipt(req, resp)
            } else {
                wrong_receipt(req, resp)
            }
        };
        let up = upstream_with(|_| "a", false, reply, sign);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY).0, 200);
        rig.wait_lines(1);
        assert_eq!(rig.post(BODY).0, 429);
        rig.wait_lines(2);
        assert_eq!(rig.post(BODY).0, 200);
        let lines = rig.wait_lines(3);
        let results: Vec<&str> = lines
            .iter()
            .map(|l| l["result"].as_str().unwrap())
            .collect();
        assert_eq!(results[0], "verified");
        assert_eq!(results[1], "not-signed (HTTP 429)");
        assert!(results[2].starts_with("failed:"), "{results:?}");
        wait_for("fatal", || rig.fatal.reason().is_some());
        no_key_in(&rig);
    }

    /// Review fix: a response still relaying when the forwarder stops gets
    /// a line and a non-verified status after the drain.
    #[test]
    fn a_check_unfinished_after_the_drain_is_recorded_as_failed() {
        let last_sent = Arc::new(Mutex::new(None));
        let sent = last_sent.clone();
        let pauses = vec![Duration::ZERO, Duration::ZERO, Duration::from_secs(5)];
        let up = upstream(|_| "a", false, move |_| sse_answer(pauses.clone(), &sent));
        let calls = Arc::new(AtomicUsize::new(0));
        let policy = SignaturePolicy {
            window: Duration::from_millis(300),
            ..quick()
        };
        let mut rig = rig_with(
            &up,
            &["a"],
            &[test_vk()],
            reattest(&["a"], &calls),
            1 << 20,
            policy,
        );
        let mut resp = post_stream(&rig);
        let mut buf = [0u8; 64];
        assert!(resp.read(&mut buf).unwrap() > 0);
        drop(resp);
        let start = Instant::now();
        rig.stop_forwarder();
        assert!(
            start.elapsed() < Duration::from_secs(4),
            "{:?}",
            start.elapsed()
        );
        let lines = rig.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["result"], "failed: unchecked at shutdown");
        assert_eq!(lines[0]["request_sha256"], sha_hex(STREAM_BODY));
        let doc = rig.doc();
        assert_eq!(doc["status"], "signature-failed");
        assert_eq!(doc["reason"], "unchecked at shutdown");
        assert_eq!(
            rig.fatal.reason().unwrap(),
            "signature-failed: unchecked at shutdown"
        );
        no_key_in(&rig);
    }

    /// Review fix: a receipt still answered 404 when the forwarder stops is
    /// never left "verified".
    #[test]
    fn a_receipt_still_missing_at_shutdown_is_not_verified() {
        let up = upstream_with(|_| "a", false, ok_reply, |_, _| (404, b"{}".to_vec()));
        let calls = Arc::new(AtomicUsize::new(0));
        let mut rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY).0, 200);
        rig.stop_forwarder();
        let lines = rig.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        let result = lines[0]["result"].as_str().unwrap();
        assert!(result.starts_with("failed:"), "{result}");
        assert_eq!(rig.doc()["status"], "signature-failed");
    }

    /// Review fix: a check that completes inside the drain is recorded as
    /// it is.
    #[test]
    fn a_check_finishing_during_the_drain_is_verified() {
        let n = Arc::new(AtomicUsize::new(0));
        let sign = move |req: &[u8], resp: &[u8]| {
            if n.fetch_add(1, Ordering::SeqCst) < 3 {
                (404, b"{}".to_vec())
            } else {
                good_receipt(req, resp)
            }
        };
        let up = upstream_with(|_| "a", false, ok_reply, sign);
        let calls = Arc::new(AtomicUsize::new(0));
        let mut rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        assert_eq!(rig.post(BODY).0, 200);
        rig.stop_forwarder();
        let lines = rig.lines();
        assert_eq!(lines.len(), 1, "{lines:?}");
        assert_eq!(lines[0]["result"], "verified", "{}", lines[0]);
        assert_eq!(up.signature_requests().len(), 4);
        assert_eq!(rig.doc()["status"], "verified");
        assert_eq!(rig.fatal.reason(), None);
    }

    /// Review fix: a line that cannot be written is fatal.
    #[test]
    fn a_failed_inference_log_write_ends_the_session() {
        let up = upstream(|_| "a", false, ok_reply);
        let calls = Arc::new(AtomicUsize::new(0));
        let rig = rig(&up, &["a"], reattest(&["a"], &calls), 1 << 20);
        // A directory where the log file should be: every append fails.
        std::fs::create_dir(rig.log_path()).unwrap();
        assert_eq!(rig.post(BODY).0, 200);
        wait_within(Duration::from_secs(3), "fatal", || {
            rig.fatal.reason().is_some()
        });
        assert_eq!(
            rig.fatal.reason().unwrap(),
            "signature-failed: cannot write inference.jsonl"
        );
        wait_for("signature-failed", || {
            rig.doc()["status"] == "signature-failed"
        });
        assert_eq!(rig.doc()["reason"], "cannot write inference.jsonl");
        assert_eq!(rig.post(BODY).0, 502);
    }
}
