//! Per-response NEAR AI receipts.
//!
//! For each completion the forwarder hashes the exact request bytes it sent
//! and the exact response bytes it relayed, takes the chat ID from the
//! response, fetches `GET /v1/signature/{id}` over the pinned client, and
//! requires an ed25519 signature by a key attested for the model over
//! `<model>:<request sha256>:<response sha256>`. Every outcome is one line of
//! `sessions/<id>/inference.jsonl`.

use std::collections::BTreeSet;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, VerifyingKey};
use serde::{Deserialize, Serialize};

use super::pin::{PinSet, Rejected};
use crate::attest::report::hex32;

/// A receipt body is small; anything larger is refused.
pub const MAX_RECEIPT_BYTES: usize = 16 << 10;
/// Bound on what is kept to find the chat ID: a non-streamed body, or an SSE
/// line or event before the first JSON event.
pub const MAX_ID_SCAN_BYTES: usize = 16 << 20;

/// As ironwire_upstream::receipt::RawReceipt (ironwire bf8b90b).
#[derive(Debug, Clone, Deserialize)]
pub struct Receipt {
    pub text: String,
    pub signature: String,
    pub signing_address: String,
    pub signing_algo: String,
    #[serde(default)]
    pub signature_kind: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exchange {
    pub model: String,
    pub chat_id: String,
    pub streamed: bool,
    pub request_sha256: [u8; 32],
    pub response_sha256: [u8; 32],
}

/// Non-empty, at most 128 bytes, `[A-Za-z0-9_-]` (ironwire receipt.rs
/// `usable_id`), so it is safe as a URL path segment.
pub fn usable_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

pub(crate) use crate::json::top_level_string;

/// `id` of a non-streamed chat completion.
pub fn chat_id_from_json(body: &[u8]) -> Option<String> {
    top_level_string(body, "id")
}

/// Finds the chat ID in an SSE stream: the `id` of the first event whose
/// `data` is JSON. Keeps at most `MAX_ID_SCAN_BYTES` of a partial line or
/// event, and nothing once that event has been seen.
#[derive(Debug, Default)]
pub struct SseId {
    line: Vec<u8>,
    data: Option<Vec<u8>>,
    id: Option<String>,
    done: bool,
}

impl SseId {
    pub fn push(&mut self, chunk: &[u8]) {
        for &b in chunk {
            if self.done {
                return;
            }
            if b == b'\n' {
                let mut line = std::mem::take(&mut self.line);
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                self.line_done(&line);
            } else {
                self.line.push(b);
                if self.line.len() + self.data.as_ref().map_or(0, Vec::len) > MAX_ID_SCAN_BYTES {
                    self.give_up();
                }
            }
        }
    }

    pub fn id(&self) -> Option<&str> {
        self.id.as_deref()
    }

    fn give_up(&mut self) {
        self.done = true;
        self.line = Vec::new();
        self.data = None;
    }

    fn line_done(&mut self, line: &[u8]) {
        if line.is_empty() {
            // End of event.
            if let Some(data) = self.data.take()
                && serde_json::from_slice::<serde::de::IgnoredAny>(&data).is_ok()
            {
                self.id = chat_id_from_json(&data);
                self.give_up();
            }
            return;
        }
        let Some(value) = line.strip_prefix(b"data:") else {
            // Comments and other fields.
            return;
        };
        let value = value.strip_prefix(b" ").unwrap_or(value);
        match &mut self.data {
            Some(data) => {
                data.push(b'\n');
                data.extend_from_slice(value);
            }
            None => self.data = Some(value.to_vec()),
        }
    }
}

/// ed25519 only; text is "<model>:<req>:<resp>" (model must equal ex.model)
/// or "<req>:<resp>"; both hashes equal; key in `keys` (the model's attested
/// keys and the attested gateway keys); verify_strict.
/// Returns the key.
pub fn check_receipt(
    r: &Receipt,
    ex: &Exchange,
    keys: &BTreeSet<[u8; 32]>,
) -> Result<[u8; 32], String> {
    if r.signing_algo != "ed25519" {
        return Err(format!(
            "receipt signing_algo is {:?}, not ed25519",
            short(&r.signing_algo)
        ));
    }
    let parts: Vec<&str> = r.text.split(':').collect();
    let (req, resp) = match parts.as_slice() {
        [model, req, resp] => {
            if *model != ex.model {
                return Err(format!(
                    "receipt text names model {:?}, the request was for {:?}",
                    short(model),
                    ex.model
                ));
            }
            (*req, *resp)
        }
        [req, resp] => (*req, *resp),
        _ => {
            return Err(format!(
                "receipt text has {} parts, expected <model>:<request>:<response>",
                parts.len()
            ));
        }
    };
    if hex32(req).ok() != Some(ex.request_sha256) {
        return Err("receipt request hash does not match the request sent".into());
    }
    if hex32(resp).ok() != Some(ex.response_sha256) {
        return Err("receipt response hash does not match the response relayed".into());
    }
    let key = receipt_key(r)?;
    if !keys.contains(&key) {
        return Err(format!(
            "receipt signing key {} is not attested for model {:?} or its gateway",
            hex::encode(key),
            ex.model
        ));
    }
    let sig = r.signature.strip_prefix("0x").unwrap_or(&r.signature);
    let sig: [u8; 64] = hex::decode(sig)
        .ok()
        .and_then(|b| b.try_into().ok())
        .ok_or("receipt signature is not 64 hex bytes")?;
    let verifying = VerifyingKey::from_bytes(&key).map_err(|_| {
        format!(
            "receipt signing key {} is not a valid ed25519 key",
            hex::encode(key)
        )
    })?;
    verifying
        .verify_strict(r.text.as_bytes(), &Signature::from_bytes(&sig))
        .map_err(|_| "receipt signature does not verify".to_string())?;
    Ok(key)
}

/// The receipt's ed25519 public key: exactly 64 hex digits, no `0x`.
pub fn receipt_key(r: &Receipt) -> Result<[u8; 32], String> {
    hex32(&r.signing_address).map_err(|e| format!("receipt signing_address: {e}"))
}

/// At most 64 characters of untrusted text, for an error message.
fn short(s: &str) -> String {
    s.chars().take(64).collect()
}

#[derive(Debug, Clone, Copy)]
pub struct SignaturePolicy {
    pub window: Duration,
    pub first_backoff: Duration,
    pub max_backoff: Duration,
    pub stream_failure: StreamFailure,
}

impl Default for SignaturePolicy {
    fn default() -> Self {
        Self {
            window: Duration::from_secs(10),
            first_backoff: Duration::from_millis(250),
            max_backoff: Duration::from_secs(2),
            stream_failure: StreamFailure::Fatal,
        }
    }
}

/// What a response stream that upstream cut mid-way means. The partial body
/// has no receipt, so it is never "verified"; the question is whether the
/// session survives it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamFailure {
    /// `signature-failed` ends the session (the launch path).
    Fatal,
    /// One `unverified` line; the client sees the broken stream and retries.
    /// For a long headless run, where the connection is pinned to the attested
    /// gateway and a cut stream is that gateway timing out, not a forgery.
    Unverified,
}

/// Why a receipt fetch ended without a receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FetchError {
    /// The pinned verifier refused upstream's certificate. The caller treats
    /// it as it treats a refusal seen by a completion: learn or `pin-failed`.
    Rejected(Rejected),
    Failed(String),
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FetchError::Rejected(r @ Rejected::Unpinned(_)) => {
                write!(f, "signature fetch: {} is not attested", r.certificate())
            }
            FetchError::Rejected(r @ Rejected::Untrusted(_)) => write!(
                f,
                "signature fetch: {} does not chain to a trusted root",
                r.certificate()
            ),
            FetchError::Failed(reason) => f.write_str(reason),
        }
    }
}

impl From<String> for FetchError {
    fn from(reason: String) -> Self {
        FetchError::Failed(reason)
    }
}

/// GET {upstream}/signature/{chat_id}?model=<id>&signing_algo=ed25519 with
/// `authorization`. 404, 429 and 5xx retry with doubling backoff until
/// `window`; other non-2xx fail at once; a body over 16 KiB or not a Receipt
/// fails. A transport error for which `pins` holds a certificate refusal
/// returns that refusal at once. Errors never include `authorization`.
pub async fn fetch_receipt(
    client: &reqwest::Client,
    upstream: &str,
    authorization: &reqwest::header::HeaderValue,
    ex: &Exchange,
    policy: SignaturePolicy,
    pins: &PinSet,
) -> Result<(Receipt, Duration), FetchError> {
    if !usable_id(&ex.chat_id) {
        return Err("signature fetch: unusable chat ID".to_string().into());
    }
    let start = Instant::now();
    let deadline = start + policy.window;
    let url = format!(
        "{}/signature/{}",
        upstream.trim_end_matches('/'),
        ex.chat_id
    );
    let mut backoff = policy.first_backoff;
    let mut last = String::from("no attempt");
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(format!(
                "no signature within {} ms (last: {last})",
                policy.window.as_millis()
            )
            .into());
        }
        let attempt = client
            .get(&url)
            .query(&[("model", ex.model.as_str()), ("signing_algo", "ed25519")])
            .header(reqwest::header::AUTHORIZATION, authorization.clone())
            .timeout(remaining)
            .send()
            .await;
        match attempt {
            Ok(resp) => {
                let status = resp.status();
                if status.is_success() {
                    let body = read_capped(resp, MAX_RECEIPT_BYTES).await?;
                    let receipt = serde_json::from_slice::<Receipt>(&body)
                        .map_err(|_| "signature fetch: the answer is not a receipt".to_string())?;
                    return Ok((receipt, start.elapsed()));
                }
                let code = status.as_u16();
                if code == 404 || code == 429 || status.is_server_error() {
                    last = format!("HTTP {code}");
                } else {
                    return Err(format!("signature fetch: HTTP {code}").into());
                }
            }
            // A refused certificate is the caller's to handle. Other
            // transport errors and timeouts retry inside the window. The
            // error text is not used: it is generic here and could only
            // carry the URL.
            Err(e) => {
                if let Some(r) = pins.take_rejected() {
                    return Err(FetchError::Rejected(r));
                }
                last = if e.is_timeout() {
                    "timed out"
                } else {
                    "request failed"
                }
                .into();
            }
        }
        let pause = backoff.min(deadline.saturating_duration_since(Instant::now()));
        tokio::time::sleep(pause).await;
        backoff = (backoff * 2).min(policy.max_backoff);
    }
}

async fn read_capped(mut resp: reqwest::Response, cap: usize) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    loop {
        match resp.chunk().await {
            Ok(Some(chunk)) => {
                if body.len() + chunk.len() > cap {
                    return Err(format!("signature fetch: receipt over {cap} bytes"));
                }
                body.extend_from_slice(&chunk);
            }
            Ok(None) => return Ok(body),
            Err(_) => return Err("signature fetch: reading the receipt failed".into()),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct InferenceLine {
    pub ts: u64,
    pub chat_id: String,
    pub model: String,
    pub streamed: bool,
    pub request_sha256: String,
    pub response_sha256: String,
    pub signing_key: Option<String>,
    /// Which attested role's key signed: "gateway" or "model". None unless
    /// verified.
    pub signer: Option<&'static str>,
    pub signature_ms: Option<u64>,
    pub result: String,
}

/// One JSON line, appended (O_APPEND) to a 0600 file.
pub fn append_inference_line(path: &Path, line: &InferenceLine) -> std::io::Result<()> {
    let mut bytes = serde_json::to_vec(line)?;
    bytes.push(b'\n');
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .mode(0o600)
        .open(path)?;
    // One write call per line.
    f.write_all(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey, Verifier};
    use sha2::{Digest, Sha256};
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    const REQ: &[u8] = br#"{"model":"m","messages":[]}"#;
    const RESP: &[u8] = br#"{"id":"chatcmpl-1","choices":[]}"#;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[9; 32])
    }

    fn sha(b: &[u8]) -> [u8; 32] {
        Sha256::digest(b).into()
    }

    fn exchange() -> Exchange {
        Exchange {
            model: "m".into(),
            chat_id: "chatcmpl-1".into(),
            streamed: false,
            request_sha256: sha(REQ),
            response_sha256: sha(RESP),
        }
    }

    fn signed(text: &str) -> Receipt {
        Receipt {
            text: text.into(),
            signature: hex::encode(key().sign(text.as_bytes()).to_bytes()),
            signing_address: hex::encode(key().verifying_key().to_bytes()),
            signing_algo: "ed25519".into(),
            signature_kind: None,
        }
    }

    fn good_text() -> String {
        format!("m:{}:{}", hex::encode(sha(REQ)), hex::encode(sha(RESP)))
    }

    fn attested() -> BTreeSet<[u8; 32]> {
        BTreeSet::from([key().verifying_key().to_bytes()])
    }

    #[test]
    fn the_chat_id_comes_from_json_or_the_first_sse_data_event() {
        assert_eq!(
            chat_id_from_json(br#"{"id":"chatcmpl-1","object":"chat.completion"}"#).as_deref(),
            Some("chatcmpl-1")
        );
        assert_eq!(chat_id_from_json(br#"{"object":"chat.completion"}"#), None);
        assert_eq!(chat_id_from_json(b"not json"), None);
        assert_eq!(chat_id_from_json(br#"{"id":7}"#), None);
        assert_eq!(chat_id_from_json(br#"{"id":"a","id":"b"}"#), None);

        let mut sse = SseId::default();
        assert_eq!(sse.id(), None);
        for chunk in [
            &b": ping\n\n"[..],
            b"data: {\"id\":\"c",
            b"1\",\"choices\":[]}\n\n",
            b"data: {\"id\":\"c2\"}\n\n",
            b"data: [DONE]\n\n",
        ] {
            sse.push(chunk);
        }
        assert_eq!(sse.id(), Some("c1"));

        // CRLF line ends, and a non-JSON event before the first JSON one.
        let mut sse = SseId::default();
        sse.push(b"data: hello\r\n\r\ndata: {\"id\":\"c3\"}\r\n\r\n");
        assert_eq!(sse.id(), Some("c3"));

        // The first JSON event decides, even without an id.
        let mut sse = SseId::default();
        sse.push(b"data: {\"x\":1}\n\ndata: {\"id\":\"c4\"}\n\n");
        assert_eq!(sse.id(), None);

        // A line that never ends stops the scan without growing past the cap.
        let mut sse = SseId::default();
        sse.push(b"data: ");
        sse.push(&vec![b'x'; MAX_ID_SCAN_BYTES + 1]);
        sse.push(b"\n\ndata: {\"id\":\"c5\"}\n\n");
        assert_eq!(sse.id(), None);
        assert!(sse.line.is_empty() && sse.data.is_none());

        assert!(!usable_id("../attestation/report"));
        assert!(usable_id("chatcmpl-abc_1"));
        assert!(!usable_id(""));
        assert!(!usable_id(&"a".repeat(129)));
        assert!(usable_id(&"a".repeat(128)));
        assert!(!usable_id("a?b"));
    }

    #[test]
    fn a_good_receipt_verifies_in_both_text_forms() {
        let want = key().verifying_key().to_bytes();
        assert_eq!(
            check_receipt(&signed(&good_text()), &exchange(), &attested()),
            Ok(want)
        );
        let two = format!("{}:{}", hex::encode(sha(REQ)), hex::encode(sha(RESP)));
        assert_eq!(
            check_receipt(&signed(&two), &exchange(), &attested()),
            Ok(want)
        );
        // A 0x-prefixed signature, as ironwire proof.rs accepts.
        let mut r = signed(&good_text());
        r.signature = format!("0x{}", r.signature);
        assert_eq!(check_receipt(&r, &exchange(), &attested()), Ok(want));
    }

    #[test]
    fn each_bad_receipt_is_refused() {
        let ex = exchange();
        let keys = attested();
        let refused = |r: &Receipt, why: &str| {
            let e = check_receipt(r, &ex, &keys).expect_err(why);
            assert!(e.contains(why), "{e:?} should mention {why:?}");
        };
        let (req, resp) = (hex::encode(sha(REQ)), hex::encode(sha(RESP)));

        refused(&signed(&format!("n:{req}:{resp}")), "model");
        let mut off = req.clone().into_bytes();
        off[10] = if off[10] == b'0' { b'1' } else { b'0' };
        let off = String::from_utf8(off).unwrap();
        refused(&signed(&format!("m:{off}:{resp}")), "request");
        refused(
            &signed(&format!("m:{req}:{}", hex::encode(sha(b"other")))),
            "response",
        );

        let mut r = signed(&good_text());
        let other = SigningKey::from_bytes(&[8; 32]);
        r.signature = hex::encode(other.sign(good_text().as_bytes()).to_bytes());
        r.signing_address = hex::encode(other.verifying_key().to_bytes());
        refused(&r, "not attested");

        let mut r = signed(&good_text());
        r.signature = signed("something else").signature;
        refused(&r, "signature");

        let mut r = signed(&good_text());
        r.signing_algo = "ecdsa".into();
        refused(&r, "ed25519");

        refused(&signed(&format!("m:x:{req}:{resp}")), "text");
        refused(&signed("m"), "text");

        let mut r = signed(&good_text());
        r.signature = "zz".repeat(64);
        assert!(check_receipt(&r, &ex, &keys).is_err());
        let mut r = signed(&good_text());
        r.signing_address = "0x".to_string() + &r.signing_address;
        assert!(check_receipt(&r, &ex, &keys).is_err());

        // The identity point as the key, with R = identity and s = 0: the
        // non-strict `verify` accepts it for any message (sB - kA = identity
        // = R); `verify_strict` refuses the small-order key. The first
        // assertion proves the vector tells the two apart.
        let mut identity = [0u8; 32];
        identity[0] = 1;
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&identity);
        let weak = VerifyingKey::from_bytes(&identity).unwrap();
        assert!(
            weak.verify(good_text().as_bytes(), &Signature::from_bytes(&sig))
                .is_ok(),
            "the vector must pass a non-strict verify"
        );
        let r = Receipt {
            text: good_text(),
            signature: hex::encode(sig),
            signing_address: hex::encode(identity),
            signing_algo: "ed25519".into(),
            signature_kind: None,
        };
        let e = check_receipt(&r, &ex, &BTreeSet::from([identity])).unwrap_err();
        assert!(e.contains("does not verify"), "{e}");
    }

    #[test]
    fn the_default_policy_is_ten_seconds_from_250_ms_to_2_s() {
        let p = SignaturePolicy::default();
        assert_eq!(p.window, Duration::from_secs(10));
        assert_eq!(p.first_backoff, Duration::from_millis(250));
        assert_eq!(p.max_backoff, Duration::from_secs(2));
    }

    /// A plain-HTTP server on 127.0.0.1 that answers the n-th request with
    /// `answers(n)` and records each request head.
    fn plain_server(
        answers: impl Fn(usize) -> (u16, Vec<u8>) + Send + 'static,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let heads = Arc::new(Mutex::new(Vec::new()));
        let seen = heads.clone();
        let n = AtomicUsize::new(0);
        std::thread::spawn(move || {
            for tcp in listener.incoming() {
                let Ok(mut tcp) = tcp else { continue };
                let mut reader = BufReader::new(tcp.try_clone().unwrap());
                let mut head = String::new();
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 {
                        break;
                    }
                    head.push_str(&line);
                    if line == "\r\n" {
                        break;
                    }
                }
                seen.lock().unwrap().push(head);
                let (status, body) = answers(n.fetch_add(1, Ordering::SeqCst));
                let _ = write!(
                    tcp,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = tcp.write_all(&body);
            }
        });
        (format!("http://127.0.0.1:{port}/v1"), heads)
    }

    fn receipt_json() -> Vec<u8> {
        let r = signed(&good_text());
        serde_json::to_vec(&serde_json::json!({
            "text": r.text, "signature": r.signature,
            "signing_address": r.signing_address, "signing_algo": r.signing_algo,
        }))
        .unwrap()
    }

    fn client() -> reqwest::Client {
        reqwest::Client::builder().no_proxy().build().unwrap()
    }

    /// Plain HTTP never consults the pins.
    fn no_pins() -> std::sync::Arc<PinSet> {
        PinSet::new(BTreeSet::new())
    }

    fn fast(window_ms: u64) -> SignaturePolicy {
        SignaturePolicy {
            window: Duration::from_millis(window_ms),
            first_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(50),
            ..Default::default()
        }
    }

    const AUTH: &str = "Bearer rts-dummy-key-never-send-to-a-provider";

    #[tokio::test]
    async fn fetch_retries_404_inside_the_window_and_fails_after_it() {
        let auth = reqwest::header::HeaderValue::from_static(AUTH);

        let (url, heads) = plain_server(|n| {
            if n < 2 {
                (404, b"{}".to_vec())
            } else {
                (200, receipt_json())
            }
        });
        let (r, _) = fetch_receipt(&client(), &url, &auth, &exchange(), fast(5000), &no_pins())
            .await
            .unwrap();
        assert_eq!(r.text, good_text());
        let heads = heads.lock().unwrap().clone();
        assert_eq!(heads.len(), 3);
        assert!(
            heads[0].starts_with(
                "GET /v1/signature/chatcmpl-1?model=m&signing_algo=ed25519 HTTP/1.1\r\n"
            ),
            "{}",
            heads[0]
        );
        assert!(
            heads[0]
                .lines()
                .any(|l| l.eq_ignore_ascii_case(&format!("authorization: {AUTH}")))
        );

        let (url, heads) = plain_server(|_| (404, b"{}".to_vec()));
        // Built outside the timed span: loading the native roots is slow
        // under host load and is not part of the window.
        let c = client();
        let start = Instant::now();
        // A 1 s window leaves room for three attempts on a loaded host; the 8 s
        // bound still fails if the 10 s production window were used instead.
        let e = fetch_receipt(&c, &url, &auth, &exchange(), fast(1000), &no_pins())
            .await
            .unwrap_err()
            .to_string();
        assert!(e.contains("no signature within"), "{e}");
        assert!(!e.contains("dummy"), "{e}");
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(8), "gave up after {elapsed:?}");
        let attempts = heads.lock().unwrap().len();
        assert!(attempts >= 3, "only {attempts} attempts inside the window");

        for status in [401, 400, 403] {
            let (url, heads) = plain_server(move |_| (status, b"{}".to_vec()));
            let e = fetch_receipt(&client(), &url, &auth, &exchange(), fast(5000), &no_pins())
                .await
                .unwrap_err()
                .to_string();
            assert!(e.contains(&status.to_string()), "{e}");
            assert!(!e.contains("dummy"), "{e}");
            assert_eq!(heads.lock().unwrap().len(), 1, "{status}");
        }

        // 429 and 5xx retry like 404.
        let (url, heads) = plain_server(|n| match n {
            0 => (429, b"{}".to_vec()),
            1 => (503, b"{}".to_vec()),
            _ => (200, receipt_json()),
        });
        assert!(
            fetch_receipt(&client(), &url, &auth, &exchange(), fast(5000), &no_pins())
                .await
                .is_ok()
        );
        assert_eq!(heads.lock().unwrap().len(), 3);

        // Too large, or not a receipt: refused at once.
        let (url, heads) = plain_server(|_| (200, vec![b' '; MAX_RECEIPT_BYTES + 1]));
        assert!(
            fetch_receipt(&client(), &url, &auth, &exchange(), fast(5000), &no_pins())
                .await
                .is_err()
        );
        assert_eq!(heads.lock().unwrap().len(), 1);
        let (url, _) = plain_server(|_| (200, br#"{"text":"x"}"#.to_vec()));
        assert!(
            fetch_receipt(&client(), &url, &auth, &exchange(), fast(5000), &no_pins())
                .await
                .is_err()
        );

        // An unusable chat ID is never put in a URL.
        let (url, heads) = plain_server(|_| (200, receipt_json()));
        let mut ex = exchange();
        ex.chat_id = "../attestation/report".into();
        assert!(
            fetch_receipt(&client(), &url, &auth, &ex, fast(5000), &no_pins())
                .await
                .is_err()
        );
        assert!(heads.lock().unwrap().is_empty());
    }

    #[test]
    fn inference_lines_are_appended_to_a_private_file() {
        use std::os::unix::fs::PermissionsExt;
        let dir = crate::testutil::tempdir("inference-log");
        let path = dir.join("inference.jsonl");
        let line = |result: &str| InferenceLine {
            ts: 1,
            chat_id: "c".into(),
            model: "m".into(),
            streamed: true,
            request_sha256: "aa".into(),
            response_sha256: "bb".into(),
            signing_key: None,
            signer: None,
            signature_ms: Some(3),
            result: result.into(),
        };
        append_inference_line(&path, &line("verified")).unwrap();
        append_inference_line(&path, &line("not-signed (HTTP 429)")).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(text.ends_with('\n'));
        assert_eq!(lines[0]["result"], "verified");
        assert_eq!(lines[0]["streamed"], true);
        assert_eq!(lines[0]["signing_key"], serde_json::Value::Null);
        assert_eq!(lines[0]["signer"], serde_json::Value::Null);
        assert_eq!(lines[1]["result"], "not-signed (HTTP 429)");
        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
