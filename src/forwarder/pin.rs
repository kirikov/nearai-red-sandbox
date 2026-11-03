//! Certificate pinning for the forwarder's upstream TLS.
//!
//! The pin lives in the rustls verifier, so it runs on every handshake and
//! nothing above TLS (connection pooling, a per-request check that a reused
//! connection skips) can bypass it. Resumption is off, so every connection
//! makes a full handshake and presents its certificate.

use std::collections::BTreeSet;
use std::sync::{Arc, Mutex, RwLock};

use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::client::{Resumption, WebPkiServerVerifier};
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{CertificateError, DigitallySignedStruct, Error, SignatureScheme};
use sha2::{Digest, Sha256};

/// SHA-256 of the certificate's SubjectPublicKeyInfo DER: the full
/// SEQUENCE (algorithm and subject public key), as
/// `openssl x509 -pubkey -noout | openssl pkey -pubin -outform DER` writes it.
/// This is what NEAR AI attests as `tls_cert_fingerprint`; it is not the hash
/// of the whole certificate. None when the certificate does not parse.
pub fn spki_sha256(cert: &CertificateDer<'_>) -> Option<[u8; 32]> {
    let ee = webpki::EndEntityCert::try_from(cert).ok()?;
    // rustls-webpki keeps only the SEQUENCE's contents and re-wraps them in
    // the SEQUENCE tag and length here.
    let spki = ee.subject_public_key_info();
    Some(Sha256::digest(spki.as_ref()).into())
}

/// Why the verifier last refused a certificate, with its SPKI SHA-256
/// fingerprint (`spki_sha256`); None when it does not parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rejected {
    /// Chains to a trusted root but is not attested: re-attestation may
    /// learn it (a rotated certificate).
    Unpinned([u8; 32]),
    /// Does not chain to a trusted root for the name, whether pinned or not.
    Untrusted(Option<[u8; 32]>),
}

impl Rejected {
    /// The refused certificate as a reason names it: by its SPKI SHA-256,
    /// or as one that does not parse.
    pub fn certificate(&self) -> String {
        let fp = match self {
            Rejected::Unpinned(fp) => Some(fp),
            Rejected::Untrusted(fp) => fp.as_ref(),
        };
        match fp {
            Some(fp) => format!(
                "upstream certificate with public key sha256 {}",
                hex::encode(fp)
            ),
            None => "upstream certificate that does not parse".to_string(),
        }
    }
}

/// The attested end-entity certificate fingerprints, each the SHA-256 of a
/// SubjectPublicKeyInfo. It only grows.
pub struct PinSet {
    pins: RwLock<BTreeSet<[u8; 32]>>,
    last: Mutex<Option<Rejected>>,
}

impl std::fmt::Debug for PinSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinSet")
            .field("pins", &self.pins.read().map(|p| p.len()).unwrap_or(0))
            .finish_non_exhaustive()
    }
}

impl PinSet {
    pub fn new(seed: BTreeSet<[u8; 32]>) -> Arc<Self> {
        Arc::new(Self {
            pins: RwLock::new(seed),
            last: Mutex::new(None),
        })
    }

    pub fn extend(&self, more: &BTreeSet<[u8; 32]>) {
        self.pins
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .extend(more.iter().copied());
    }

    pub fn contains(&self, fp: &[u8; 32]) -> bool {
        self.pins
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(fp)
    }

    /// Takes the last rejection, if any.
    pub fn take_rejected(&self) -> Option<Rejected> {
        self.last.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Takes the last rejection; Some only when it was an unpinned,
    /// otherwise trusted certificate.
    pub fn take_unknown(&self) -> Option<[u8; 32]> {
        match self.take_rejected()? {
            Rejected::Unpinned(fp) => Some(fp),
            Rejected::Untrusted(_) => None,
        }
    }

    fn reject(&self, r: Rejected) {
        *self.last.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    }
}

/// WebPKI first (chain to the roots, name, validity), then the pin.
/// Handshake signature checks delegate to WebPKI.
#[derive(Debug)]
pub struct PinnedVerifier {
    webpki: Arc<WebPkiServerVerifier>,
    pins: Arc<PinSet>,
}

impl PinnedVerifier {
    pub fn new(roots: Arc<rustls::RootCertStore>, pins: Arc<PinSet>) -> Result<Self, String> {
        let webpki = WebPkiServerVerifier::builder_with_provider(roots, Arc::new(provider()))
            .build()
            .map_err(|e| format!("forwarder: certificate verifier: {e}"))?;
        Ok(Self { webpki, pins })
    }
}

impl ServerCertVerifier for PinnedVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, Error> {
        let fp = spki_sha256(end_entity);
        if let Err(e) = self.webpki.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        ) {
            self.pins.reject(Rejected::Untrusted(fp));
            return Err(e);
        }
        // WebPKI parsed it, so this is unreachable; refuse rather than assume.
        let Some(fp) = fp else {
            self.pins.reject(Rejected::Untrusted(None));
            return Err(Error::InvalidCertificate(CertificateError::BadEncoding));
        };
        if !self.pins.contains(&fp) {
            self.pins.reject(Rejected::Unpinned(fp));
            return Err(Error::InvalidCertificate(
                CertificateError::ApplicationVerificationFailure,
            ));
        }
        Ok(ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        self.webpki.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        self.webpki.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.webpki.supported_verify_schemes()
    }
}

fn provider() -> rustls::crypto::CryptoProvider {
    rustls::crypto::ring::default_provider()
}

/// The operating system's trust roots; an empty store is an error.
pub fn native_roots() -> Result<Arc<rustls::RootCertStore>, String> {
    let loaded = rustls_native_certs::load_native_certs();
    let mut store = rustls::RootCertStore::empty();
    store.add_parsable_certificates(loaded.certs);
    if store.is_empty() {
        return Err(format!(
            "forwarder: no usable system trust roots ({} load errors)",
            loaded.errors.len()
        ));
    }
    Ok(Arc::new(store))
}

/// ring provider, PinnedVerifier, `resumption = Resumption::disabled()`, no ALPN (HTTP/1.1).
/// Ordinary TLS to the real upstream host: the system trust roots verify the
/// certificate chains to a public CA for the host name, with no pinning. The
/// attestation binding then comes from the per-response signatures, not the
/// connection, so the gateway may fail over across instances (it shares one
/// signing key per model). Used by the signature-attested forwarder mode.
pub fn native_tls(roots: Arc<rustls::RootCertStore>) -> Result<rustls::ClientConfig, String> {
    let mut cfg = rustls::ClientConfig::builder_with_provider(Arc::new(provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("forwarder: TLS versions: {e}"))?
        .with_root_certificates((*roots).clone())
        .with_no_client_auth();
    cfg.resumption = Resumption::disabled();
    cfg.alpn_protocols = Vec::new();
    cfg.enable_early_data = false;
    Ok(cfg)
}

pub fn pinned_tls(
    roots: Arc<rustls::RootCertStore>,
    pins: Arc<PinSet>,
) -> Result<rustls::ClientConfig, String> {
    let verifier = PinnedVerifier::new(roots, pins)?;
    let mut cfg = rustls::ClientConfig::builder_with_provider(Arc::new(provider()))
        .with_safe_default_protocol_versions()
        .map_err(|e| format!("forwarder: TLS versions: {e}"))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(verifier))
        .with_no_client_auth();
    cfg.resumption = Resumption::disabled();
    cfg.alpn_protocols = Vec::new();
    cfg.enable_early_data = false;
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustls::client::danger::ServerCertVerifier;
    use rustls::pki_types::pem::PemObject;
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use std::collections::BTreeSet;
    use std::sync::Arc;

    fn cert(pem: &[u8]) -> CertificateDer<'static> {
        CertificateDer::from_pem_slice(pem).unwrap()
    }

    fn roots() -> Arc<rustls::RootCertStore> {
        let mut store = rustls::RootCertStore::empty();
        store
            .add(cert(include_bytes!("../../tests/fixtures/tls/ca.pem")))
            .unwrap();
        Arc::new(store)
    }

    fn spki(c: &CertificateDer<'_>) -> [u8; 32] {
        spki_sha256(c).expect("parses")
    }

    /// The live cloud-api.near.ai leaf, taken from the production report's
    /// `tls_certificate` on 2026-10-29. NEAR AI's gateway attested
    /// `tls_cert_fingerprint` 73862c25... for it, which is sha256 of its
    /// SubjectPublicKeyInfo:
    /// `openssl x509 -pubkey -noout | openssl pkey -pubin -outform DER | shasum -a 256`.
    #[test]
    fn the_production_leaf_hashes_to_its_attested_spki_fingerprint() {
        let leaf = cert(include_bytes!(
            "../../tests/fixtures/near-prod/cloud-api-leaf.pem"
        ));
        assert_eq!(
            hex::encode(spki(&leaf)),
            "73862c254372091cc6895b900111503860694f8916ebf0c63b18d245ba8ba5ad"
        );
    }

    /// Regression: the pin is not sha256 of the whole leaf DER. That hash
    /// (5664...) is what the forwarder used to compute, and it never
    /// matches what production attests.
    #[test]
    fn the_leaf_der_hash_is_not_the_pin() {
        let leaf = cert(include_bytes!(
            "../../tests/fixtures/near-prod/cloud-api-leaf.pem"
        ));
        let der: [u8; 32] = Sha256::digest(leaf.as_ref()).into();
        assert_eq!(
            hex::encode(der),
            "5664580137f5b2abc8b2cc9ef53dc2bd7cd79b6ffc9f840dbbfca708a2c51022"
        );
        assert_ne!(spki(&leaf), der);

        // A verifier pinned to the leaf-DER hash refuses the certificate.
        let pins = PinSet::new(BTreeSet::from([der]));
        let v = PinnedVerifier::new(roots(), pins.clone()).unwrap();
        let a = cert(include_bytes!("../../tests/fixtures/tls/a.pem"));
        pins.extend(&BTreeSet::from([Sha256::digest(a.as_ref()).into()]));
        let name = ServerName::try_from("127.0.0.1").unwrap();
        assert!(
            v.verify_server_cert(&a, &[], &name, &[], UnixTime::now())
                .is_err()
        );
        assert_eq!(pins.take_rejected(), Some(Rejected::Unpinned(spki(&a))));
    }

    /// The openssl-generated test certificates, fingerprinted by SPKI as
    /// openssl computes it.
    #[test]
    fn the_test_certificates_are_fingerprinted_by_spki() {
        for (pem, want) in [
            (
                &include_bytes!("../../tests/fixtures/tls/a.pem")[..],
                "37e84e0fb98ff9dd3644b50b55b1c95051550e4dbe14d3410ef9b77e31573b80",
            ),
            (
                &include_bytes!("../../tests/fixtures/tls/b.pem")[..],
                "95426bdf061eaee3c2680eb1183f5b6e5e54096f4bc18c6889ffb85f28afce2a",
            ),
            (
                &include_bytes!("../../tests/fixtures/tls/c.pem")[..],
                "b21ddeb494ff2a89e1e2144ae531bdf17f409795c32c21f3f24db680ba239359",
            ),
        ] {
            assert_eq!(hex::encode(spki(&cert(pem))), want);
        }
    }

    /// Bytes that are not a certificate have no fingerprint, and the
    /// verifier refuses them as untrusted.
    #[test]
    fn a_certificate_that_does_not_parse_is_untrusted() {
        let junk = CertificateDer::from(b"not a certificate".to_vec());
        assert_eq!(spki_sha256(&junk), None);
        let pins = PinSet::new(BTreeSet::from([[1; 32]]));
        let v = PinnedVerifier::new(roots(), pins.clone()).unwrap();
        let name = ServerName::try_from("127.0.0.1").unwrap();
        assert!(
            v.verify_server_cert(&junk, &[], &name, &[], UnixTime::now())
                .is_err()
        );
        assert_eq!(pins.take_rejected(), Some(Rejected::Untrusted(None)));
    }

    #[test]
    fn the_pin_set_grows_and_reports_the_last_rejection_once() {
        let pins = PinSet::new(BTreeSet::from([[1; 32]]));
        assert!(pins.contains(&[1; 32]));
        assert!(!pins.contains(&[2; 32]));
        pins.extend(&BTreeSet::from([[2; 32]]));
        assert!(pins.contains(&[1; 32]) && pins.contains(&[2; 32]));
        assert_eq!(pins.take_unknown(), None);
        assert!(!format!("{pins:?}").is_empty());
    }

    #[test]
    fn the_verifier_checks_the_chain_and_then_the_pin() {
        let a = cert(include_bytes!("../../tests/fixtures/tls/a.pem"));
        let b = cert(include_bytes!("../../tests/fixtures/tls/b.pem"));
        let c = cert(include_bytes!("../../tests/fixtures/tls/c.pem"));
        let pins = PinSet::new(BTreeSet::from([spki(&a), spki(&c)]));
        let v = PinnedVerifier::new(roots(), pins.clone()).unwrap();
        let name = ServerName::try_from("127.0.0.1").unwrap();
        let now = UnixTime::now();
        assert!(v.verify_server_cert(&a, &[], &name, &[], now).is_ok());
        assert_eq!(pins.take_rejected(), None);

        assert!(v.verify_server_cert(&b, &[], &name, &[], now).is_err());
        assert_eq!(pins.take_rejected(), Some(Rejected::Unpinned(spki(&b))));
        assert_eq!(pins.take_rejected(), None, "taken once");

        assert!(v.verify_server_cert(&b, &[], &name, &[], now).is_err());
        assert_eq!(pins.take_unknown(), Some(spki(&b)));

        // Pinned, but self-signed: the chain check runs first and fails.
        assert!(v.verify_server_cert(&c, &[], &name, &[], now).is_err());
        assert_eq!(
            pins.take_rejected(),
            Some(Rejected::Untrusted(Some(spki(&c))))
        );
        assert!(v.verify_server_cert(&c, &[], &name, &[], now).is_err());
        assert_eq!(pins.take_unknown(), None, "untrusted is not unknown");

        // The right certificate for the wrong name.
        let other = ServerName::try_from("cloud-api.near.ai").unwrap();
        assert!(v.verify_server_cert(&a, &[], &other, &[], now).is_err());

        pins.extend(&BTreeSet::from([spki(&b)]));
        assert!(v.verify_server_cert(&b, &[], &name, &[], now).is_ok());
    }

    #[test]
    fn the_client_config_never_resumes_and_offers_no_alpn() {
        let cfg = pinned_tls(roots(), PinSet::new(BTreeSet::new())).unwrap();
        assert!(cfg.alpn_protocols.is_empty());
        assert!(!cfg.enable_early_data);
        assert_eq!(
            format!("{:?}", cfg.resumption),
            format!("{:?}", rustls::client::Resumption::disabled())
        );
    }

    #[test]
    fn native_roots_load() {
        assert!(!native_roots().unwrap().is_empty());
    }
}
