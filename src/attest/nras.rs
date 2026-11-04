//! NVIDIA Remote Attestation Service (NRAS) GPU verification.
//!
//! The model's GPU evidence goes to NRAS, which answers with a JWT verdict.
//! Unlike the upstream TypeScript verifier, the verdict is trusted only after
//! its ES384 signature verifies against NVIDIA's published JWKS, it has not
//! expired, and its overall result is the JSON boolean `true`.

use std::sync::Arc;

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ring::signature::{ECDSA_P384_SHA384_FIXED, UnparsedPublicKey};
use serde_json::Value;

use crate::attest::http::{Clock, Http};

pub const NRAS_ATTEST_URL: &str = "https://nras.attestation.nvidia.com/v3/attest/gpu";
pub const NRAS_JWKS_URL: &str = "https://nras.attestation.nvidia.com/.well-known/jwks.json";

/// The claim carrying NRAS's overall verdict.
const OVERALL_CLAIM: &str = "x-nvidia-overall-att-result";

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct GpuVerdict {
    pub kid: String,
    pub overall: bool,
}

fn b64(part: &str, what: &str) -> Result<Vec<u8>, String> {
    // URL_SAFE_NO_PAD refuses padding and non-canonical trailing bits.
    URL_SAFE_NO_PAD
        .decode(part)
        .map_err(|e| format!("NRAS token: {what} is not base64url: {e}"))
}

fn json_object(bytes: &[u8], what: &str) -> Result<serde_json::Map<String, Value>, String> {
    match serde_json::from_slice(bytes) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => Err(format!("NRAS token: {what} is not a JSON object")),
        Err(e) => Err(format!("NRAS token: {what} is not JSON: {e}")),
    }
}

/// The uncompressed SEC1 point `04 || x || y` of the one JWKS key named `kid`,
/// which must be `kty EC`, `crv P-384`, and, where stated, for ES384 signing.
fn jwks_point(jwks: &[u8], kid: &str) -> Result<Vec<u8>, String> {
    let doc: Value =
        serde_json::from_slice(jwks).map_err(|e| format!("NRAS JWKS is not JSON: {e}"))?;
    let keys = doc
        .get("keys")
        .and_then(Value::as_array)
        .ok_or("NRAS JWKS has no keys array")?;
    let mut found = keys
        .iter()
        .filter(|k| k.get("kid").and_then(Value::as_str) == Some(kid));
    let key = found
        .next()
        .ok_or_else(|| format!("NRAS JWKS has no key with the token's kid {kid:?}"))?;
    if found.next().is_some() {
        return Err(format!("NRAS JWKS has more than one key with kid {kid:?}"));
    }
    let field = |name: &str| key.get(name).and_then(Value::as_str);
    if field("kty") != Some("EC") {
        return Err(format!("NRAS JWKS key {kid:?} is not kty EC"));
    }
    if field("crv") != Some("P-384") {
        return Err(format!("NRAS JWKS key {kid:?} is not crv P-384"));
    }
    if key.get("alg").is_some() && field("alg") != Some("ES384") {
        return Err(format!("NRAS JWKS key {kid:?} is not for alg ES384"));
    }
    if key.get("use").is_some() && field("use") != Some("sig") {
        return Err(format!("NRAS JWKS key {kid:?} is not for use sig"));
    }
    let coord = |name: &str| -> Result<Vec<u8>, String> {
        let s = field(name).ok_or_else(|| format!("NRAS JWKS key {kid:?} has no {name}"))?;
        let v = URL_SAFE_NO_PAD
            .decode(s)
            .map_err(|e| format!("NRAS JWKS key {kid:?}: {name} is not base64url: {e}"))?;
        if v.len() != 48 {
            return Err(format!("NRAS JWKS key {kid:?}: {name} is not 48 bytes"));
        }
        Ok(v)
    };
    let mut point = Vec::with_capacity(97);
    point.push(0x04);
    point.extend(coord("x")?);
    point.extend(coord("y")?);
    Ok(point)
}

/// Verifies `alg == "ES384"`, finds the JWKS key with the token's `kid` (`kty EC`, `crv P-384`),
/// verifies the 96-byte fixed signature over `header.payload` against `04 || x || y`,
/// and requires `exp > now_unix` and `nbf <= now_unix + CLOCK_LEEWAY_SECS`. Returns the kid
/// and the claims.
/// Clock skew tolerated on `nbf` (not on `exp`).
pub const CLOCK_LEEWAY_SECS: u64 = 120;

pub fn verify_es384_jwt(
    token: &str,
    jwks: &[u8],
    now_unix: u64,
) -> Result<(String, serde_json::Value), String> {
    let parts: Vec<&str> = token.split('.').collect();
    let [h, c, s] = parts[..] else {
        return Err("NRAS token: not three dot-separated parts".into());
    };
    let header = json_object(&b64(h, "header")?, "header")?;
    match header.get("alg").and_then(Value::as_str) {
        Some("ES384") => {}
        _ => return Err("NRAS token: alg is not ES384".into()),
    }
    if header.contains_key("crit") {
        return Err("NRAS token: header carries crit extensions".into());
    }
    let kid = header
        .get("kid")
        .and_then(Value::as_str)
        .ok_or("NRAS token: header has no kid")?
        .to_string();
    let point = jwks_point(jwks, &kid)?;
    let sig = b64(s, "signature")?;
    if sig.len() != 96 {
        return Err("NRAS token: signature is not 96 bytes".into());
    }
    // The signing input is the header and payload exactly as received.
    let signed = &token[..h.len() + 1 + c.len()];
    UnparsedPublicKey::new(&ECDSA_P384_SHA384_FIXED, &point)
        .verify(signed.as_bytes(), &sig)
        .map_err(|_| format!("NRAS token: signature does not verify under kid {kid:?}"))?;
    let claims = json_object(&b64(c, "payload")?, "payload")?;
    match claims.get("exp").map(Value::as_u64) {
        Some(Some(exp)) if exp > now_unix => {}
        Some(Some(_)) => return Err("NRAS token: expired (exp has passed)".into()),
        _ => return Err("NRAS token: exp is missing or not a non-negative integer".into()),
    }
    // NRAS issues nbf == iat, so a verifier whose clock runs a few seconds
    // behind NVIDIA's would refuse every fresh token without a leeway (RFC 7519
    // §4.1.5 allows one); exp keeps no leeway, a token is never accepted late.
    match claims.get("nbf").map(Value::as_u64) {
        None => {}
        Some(Some(nbf)) if nbf <= now_unix + CLOCK_LEEWAY_SECS => {}
        Some(_) => return Err("NRAS token: nbf is in the future or not an integer".into()),
    }
    Ok((kid, Value::Object(claims)))
}

pub trait GpuVerifier: Send + Sync {
    fn verify(&self, nvidia_payload: &str, nonce_hex: &str) -> Result<GpuVerdict, String>;
}

pub struct NrasGpuVerifier {
    pub http: Arc<dyn Http>,
    pub clock: Arc<dyn Clock>,
}

impl GpuVerifier for NrasGpuVerifier {
    fn verify(&self, nvidia_payload: &str, nonce_hex: &str) -> Result<GpuVerdict, String> {
        // The nonce binds the GPU evidence to this attestation; check it before
        // anything leaves the host.
        if nonce_hex.is_empty() || !nonce_hex.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("GPU evidence: expected nonce is not hex".into());
        }
        // One top-level string "nonce": a repeated key is refused, since
        // NRAS may read the first where serde_json's Value keeps the last.
        let nonce = crate::json::top_level_string(nvidia_payload.as_bytes(), "nonce")
            .ok_or("GPU evidence is not a JSON object with exactly one string nonce")?;
        if !nonce.eq_ignore_ascii_case(nonce_hex) {
            return Err("GPU evidence nonce does not match the request nonce".into());
        }

        let resp = self
            .http
            .post_json(NRAS_ATTEST_URL, nvidia_payload.as_bytes())
            .map_err(|e| format!("NRAS attest: {e}"))?;
        if resp.status != 200 {
            return Err(format!("NRAS attest returned status {}", resp.status));
        }
        let answer: Value = serde_json::from_slice(&resp.body)
            .map_err(|e| format!("NRAS attest answer is not JSON: {e}"))?;
        let token = answer
            .get(0)
            .and_then(|e| e.get(1))
            .and_then(Value::as_str)
            .ok_or("NRAS attest answer has no overall token at [0][1]")?;

        // Fetched each time: NVIDIA rotates its signing keys daily.
        let jwks = self
            .http
            .get(NRAS_JWKS_URL, None)
            .map_err(|e| format!("NRAS JWKS: {e}"))?;
        if jwks.status != 200 {
            return Err(format!("NRAS JWKS returned status {}", jwks.status));
        }
        let (kid, claims) = verify_es384_jwt(token, &jwks.body, self.clock.now_unix())?;
        if claims.get(OVERALL_CLAIM) != Some(&Value::Bool(true)) {
            return Err(format!("NRAS verdict: {OVERALL_CLAIM} is not true"));
        }
        Ok(GpuVerdict { kid, overall: true })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CLOCK_LEEWAY_SECS, GpuVerdict, GpuVerifier, NRAS_ATTEST_URL, NRAS_JWKS_URL,
        NrasGpuVerifier, verify_es384_jwt,
    };
    use crate::attest::http::{FakeHttp, FixedClock, HttpResponse};
    use base64::Engine;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use ring::rand::SystemRandom;
    use ring::signature::{ECDSA_P384_SHA384_FIXED_SIGNING, EcdsaKeyPair, KeyPair};
    use serde_json::{Value, json};
    use std::sync::Arc;

    const A: &[u8] = include_bytes!("../../tests/fixtures/nras/p384-a.pk8");
    const B: &[u8] = include_bytes!("../../tests/fixtures/nras/p384-b.pk8");
    const NOW: u64 = 1_793_000_000;

    fn nonce() -> String {
        "5f".repeat(32)
    }

    fn key(pk8: &[u8]) -> EcdsaKeyPair {
        EcdsaKeyPair::from_pkcs8(&ECDSA_P384_SHA384_FIXED_SIGNING, pk8, &SystemRandom::new())
            .unwrap()
    }

    fn jwks_crv(kid: &str, k: &EcdsaKeyPair, crv: &str) -> Vec<u8> {
        let pk = k.public_key().as_ref();
        assert_eq!(pk.len(), 97);
        serde_json::to_vec(&json!({"keys": [{
            "kty": "EC",
            "crv": crv,
            "kid": kid,
            "x": URL_SAFE_NO_PAD.encode(&pk[1..49]),
            "y": URL_SAFE_NO_PAD.encode(&pk[49..97]),
            "x5c": ["test"],
        }]}))
        .unwrap()
    }

    fn jwks(kid: &str, k: &EcdsaKeyPair) -> Vec<u8> {
        jwks_crv(kid, k, "P-384")
    }

    fn token(k: &EcdsaKeyPair, header: Value, claims: Value) -> String {
        let h = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&header).unwrap());
        let c = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap());
        let input = format!("{h}.{c}");
        let sig = k.sign(&SystemRandom::new(), input.as_bytes()).unwrap();
        format!("{input}.{}", URL_SAFE_NO_PAD.encode(sig.as_ref()))
    }

    fn header() -> Value {
        json!({"alg": "ES384", "kid": "k1", "typ": "JWT"})
    }

    fn claims(verdict: bool) -> Value {
        json!({"exp": NOW + 60, "x-nvidia-overall-att-result": verdict})
    }

    #[test]
    fn a_token_signed_by_the_jwks_key_verifies() {
        let a = key(A);
        let t = token(&a, header(), claims(true));
        let (kid, got) = verify_es384_jwt(&t, &jwks("k1", &a), NOW).unwrap();
        assert_eq!(kid, "k1");
        assert_eq!(got, claims(true));
    }

    #[test]
    fn a_token_signed_by_another_key_is_refused() {
        let t = token(&key(B), header(), claims(true));
        let e = verify_es384_jwt(&t, &jwks("k1", &key(A)), NOW).unwrap_err();
        assert!(e.contains("signature"), "{e}");
    }

    #[test]
    fn an_unknown_kid_is_refused() {
        let a = key(A);
        let t = token(&a, header(), claims(true));
        let e = verify_es384_jwt(&t, &jwks("k2", &a), NOW).unwrap_err();
        assert!(e.contains("kid"), "{e}");
        let t = token(&a, json!({"alg": "ES384", "typ": "JWT"}), claims(true));
        let e = verify_es384_jwt(&t, &jwks("k1", &a), NOW).unwrap_err();
        assert!(e.contains("kid"), "{e}");
    }

    #[test]
    fn any_alg_but_es384_is_refused() {
        let a = key(A);
        for alg in ["ES256", "none", "HS384", "es384"] {
            let t = token(
                &a,
                json!({"alg": alg, "kid": "k1", "typ": "JWT"}),
                claims(true),
            );
            let e = verify_es384_jwt(&t, &jwks("k1", &a), NOW).unwrap_err();
            assert!(e.contains("alg"), "{alg}: {e}");
        }
        let t = token(&a, json!({"kid": "k1"}), claims(true));
        let e = verify_es384_jwt(&t, &jwks("k1", &a), NOW).unwrap_err();
        assert!(e.contains("alg"), "{e}");
    }

    #[test]
    fn an_edited_payload_is_refused() {
        let a = key(A);
        let t = token(&a, header(), claims(false));
        let other = URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims(true)).unwrap());
        let parts: Vec<&str> = t.split('.').collect();
        let forged = format!("{}.{}.{}", parts[0], other, parts[2]);
        assert!(verify_es384_jwt(&forged, &jwks("k1", &a), NOW).is_err());
        // Malformed shapes are refused too.
        for bad in [
            String::new(),
            format!("{}.{}", parts[0], parts[1]),
            format!("{t}.extra"),
            format!("{}.{}.{}=", parts[0], parts[1], parts[2]),
        ] {
            assert!(
                verify_es384_jwt(&bad, &jwks("k1", &a), NOW).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn an_expired_or_unexpiring_token_is_refused() {
        let a = key(A);
        for c in [
            json!({"exp": NOW - 1, "x-nvidia-overall-att-result": true}),
            json!({"exp": NOW, "x-nvidia-overall-att-result": true}),
            json!({"x-nvidia-overall-att-result": true}),
            json!({"exp": "9999999999", "x-nvidia-overall-att-result": true}),
        ] {
            let t = token(&a, header(), c.clone());
            let e = verify_es384_jwt(&t, &jwks("k1", &a), NOW).unwrap_err();
            assert!(e.contains("exp"), "{c}: {e}");
        }
    }

    #[test]
    fn nbf_tolerates_the_clock_leeway_but_not_more() {
        let a = key(A);
        for (nbf, accepted) in [
            (NOW, true),
            (NOW + CLOCK_LEEWAY_SECS, true),
            (NOW + CLOCK_LEEWAY_SECS + 1, false),
        ] {
            let c = json!({"exp": NOW + 3600, "nbf": nbf, "x-nvidia-overall-att-result": true});
            let r = verify_es384_jwt(&token(&a, header(), c), &jwks("k1", &a), NOW);
            assert_eq!(r.is_ok(), accepted, "nbf = NOW + {}: {r:?}", nbf - NOW);
            if !accepted {
                assert!(r.unwrap_err().contains("nbf"));
            }
        }
        let c = json!({"exp": NOW + 3600, "nbf": "soon", "x-nvidia-overall-att-result": true});
        assert!(verify_es384_jwt(&token(&a, header(), c), &jwks("k1", &a), NOW).is_err());
    }

    #[test]
    fn a_jwks_key_on_another_curve_is_refused() {
        let a = key(A);
        let t = token(&a, header(), claims(true));
        assert!(verify_es384_jwt(&t, &jwks_crv("k1", &a, "P-256"), NOW).is_err());
        let mut j: Value = serde_json::from_slice(&jwks("k1", &a)).unwrap();
        j["keys"][0]["kty"] = json!("RSA");
        assert!(verify_es384_jwt(&t, &serde_json::to_vec(&j).unwrap(), NOW).is_err());
    }

    fn ok(body: Vec<u8>) -> HttpResponse {
        HttpResponse {
            status: 200,
            headers: vec![],
            body,
        }
    }

    fn payload() -> String {
        format!(
            r#"{{"nonce":"{}","arch":"HOPPER","evidence_list":[]}}"#,
            nonce()
        )
    }

    fn verifier(verdict: bool, status: u16) -> (Arc<FakeHttp>, NrasGpuVerifier) {
        let a = key(A);
        let t = token(&a, header(), claims(verdict));
        let http = FakeHttp::new();
        http.route(
            "POST",
            NRAS_ATTEST_URL,
            HttpResponse {
                status,
                headers: vec![],
                body: serde_json::to_vec(&json!([["JWT", t]])).unwrap(),
            },
        );
        http.route("GET", NRAS_JWKS_URL, ok(jwks("k1", &a)));
        let v = NrasGpuVerifier {
            http: http.clone(),
            clock: Arc::new(FixedClock(NOW)),
        };
        (http, v)
    }

    #[test]
    fn the_verifier_posts_the_payload_verbatim_and_needs_a_true_verdict() {
        let (http, v) = verifier(true, 200);
        let got = v.verify(&payload(), &nonce().to_uppercase()).unwrap();
        assert_eq!(
            got,
            GpuVerdict {
                kid: "k1".into(),
                overall: true
            }
        );
        assert_eq!(http.bodies(), vec![payload().into_bytes()]);
        assert_eq!(
            http.calls(),
            vec![
                format!("POST {NRAS_ATTEST_URL}"),
                format!("GET {NRAS_JWKS_URL}")
            ]
        );
        assert!(http.bearers().iter().all(|b| !b));

        let (_, v) = verifier(false, 200);
        let e = v.verify(&payload(), &nonce()).unwrap_err();
        assert!(e.contains("x-nvidia-overall-att-result"), "{e}");

        let (_, v) = verifier(true, 500);
        let e = v.verify(&payload(), &nonce()).unwrap_err();
        assert!(e.contains("NRAS"), "{e}");
    }

    #[test]
    fn a_verdict_that_is_not_the_boolean_true_is_refused() {
        let a = key(A);
        for verdict in [json!("true"), json!(1), Value::Null] {
            let t = token(
                &a,
                header(),
                json!({"exp": NOW + 60, "x-nvidia-overall-att-result": verdict}),
            );
            let http = FakeHttp::new();
            http.route(
                "POST",
                NRAS_ATTEST_URL,
                ok(serde_json::to_vec(&json!([["JWT", t]])).unwrap()),
            );
            http.route("GET", NRAS_JWKS_URL, ok(jwks("k1", &a)));
            let v = NrasGpuVerifier {
                http,
                clock: Arc::new(FixedClock(NOW)),
            };
            let e = v.verify(&payload(), &nonce()).unwrap_err();
            assert!(e.contains("x-nvidia-overall-att-result"), "{verdict}: {e}");
        }
        // A token missing the claim entirely.
        let t = token(&a, header(), json!({"exp": NOW + 60}));
        let http = FakeHttp::new();
        http.route(
            "POST",
            NRAS_ATTEST_URL,
            ok(serde_json::to_vec(&json!([["JWT", t]])).unwrap()),
        );
        http.route("GET", NRAS_JWKS_URL, ok(jwks("k1", &a)));
        let v = NrasGpuVerifier {
            http,
            clock: Arc::new(FixedClock(NOW)),
        };
        assert!(v.verify(&payload(), &nonce()).is_err());
    }

    #[test]
    fn a_malformed_nras_answer_is_refused() {
        for body in [
            b"{}".to_vec(),
            b"[]".to_vec(),
            b"[[\"JWT\"]]".to_vec(),
            b"not json".to_vec(),
        ] {
            let http = FakeHttp::new();
            http.route("POST", NRAS_ATTEST_URL, ok(body.clone()));
            http.route("GET", NRAS_JWKS_URL, ok(jwks("k1", &key(A))));
            let v = NrasGpuVerifier {
                http,
                clock: Arc::new(FixedClock(NOW)),
            };
            assert!(v.verify(&payload(), &nonce()).is_err(), "{body:?}");
        }
    }

    #[test]
    fn a_payload_for_another_nonce_is_refused_before_any_request() {
        let (http, v) = verifier(true, 200);
        let e = v.verify(&payload(), &"6a".repeat(32)).unwrap_err();
        assert!(e.contains("nonce"), "{e}");
        let e = v.verify(r#"{"arch":"HOPPER"}"#, &nonce()).unwrap_err();
        assert!(e.contains("nonce"), "{e}");
        let e = v.verify("not json", &nonce()).unwrap_err();
        assert!(!e.is_empty());
        assert!(http.calls().is_empty());
    }

    /// Review finding: the nonce is the only binding of the GPU evidence to
    /// this session, and parsers disagree on which duplicate key wins, so a
    /// second "nonce" is refused whichever one matches.
    #[test]
    fn a_payload_with_two_nonces_is_refused_before_any_request() {
        let (http, v) = verifier(true, 200);
        let other = "6a".repeat(32);
        for (first, second) in [(nonce(), other.clone()), (other.clone(), nonce())] {
            let p = format!(
                r#"{{"nonce":"{first}","arch":"HOPPER","nonce":"{second}","evidence_list":[]}}"#
            );
            let e = v.verify(&p, &nonce()).unwrap_err();
            assert!(e.contains("nonce"), "{e}");
        }
        // A non-string nonce is no nonce.
        let e = v.verify(r#"{"nonce":5}"#, &nonce()).unwrap_err();
        assert!(e.contains("nonce"), "{e}");
        assert!(http.calls().is_empty());
    }
}
