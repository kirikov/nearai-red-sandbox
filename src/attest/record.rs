//! `sessions/<id>/attestation.json`: the session's attestation evidence and
//! status, rewritten atomically (temp file in the same directory, then
//! rename) on every change.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use super::{AttestOptions, Attested};

const FILE_NAME: &str = "attestation.json";

pub struct AttestationRecord {
    path: PathBuf,
    doc: Mutex<Value>,
}

fn hex_set<'a>(set: impl IntoIterator<Item = &'a [u8; 32]>) -> Value {
    Value::Array(set.into_iter().map(|b| hex::encode(b).into()).collect())
}

/// Fingerprints, per-model keys and gateway keys, as hex.
fn verified_state(a: &Attested) -> (Value, Value, Value) {
    let keys = a
        .keys
        .iter()
        .map(|(model, keys)| (model.clone(), hex_set(keys)))
        .collect::<serde_json::Map<_, _>>();
    (
        hex_set(&a.fingerprints),
        Value::Object(keys),
        hex_set(&a.gateway_keys),
    )
}

impl AttestationRecord {
    /// Creates sessions/<id>/ (0700) and attestation.json (0600) with
    /// {"session": id, "status": "pending", "overrides": options}.
    pub fn create(
        sessions_dir: &Path,
        session_id: &str,
        options: &AttestOptions,
    ) -> Result<Arc<Self>, String> {
        if session_id.is_empty()
            || !session_id
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            return Err(format!(
                "attestation record: invalid session ID {session_id:?}"
            ));
        }
        let dir = sessions_dir.join(session_id);
        fs::DirBuilder::new()
            .recursive(true)
            .mode(0o700)
            .create(&dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700))
            .map_err(|e| format!("cannot restrict {}: {e}", dir.display()))?;
        let overrides = serde_json::to_value(options)
            .map_err(|e| format!("attestation record: overrides: {e}"))?;
        let record = Arc::new(Self {
            path: dir.join(FILE_NAME),
            doc: Mutex::new(json!({
                "session": session_id,
                "status": "pending",
                "overrides": overrides,
            })),
        });
        record.update(|_| {})?;
        Ok(record)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn session_dir(&self) -> &Path {
        self.path
            .parent()
            .expect("attestation.json lives in the session directory")
    }

    /// Status "verified", fingerprints, model keys and gateway keys as hex,
    /// and the evidence.
    pub fn verified(&self, a: &Attested) -> Result<(), String> {
        let (fingerprints, keys, gateway_keys) = verified_state(a);
        self.update(|doc| {
            doc["status"] = "verified".into();
            doc["fingerprints"] = fingerprints;
            doc["keys"] = keys;
            doc["gateway_keys"] = gateway_keys;
            doc["evidence"] = Value::Array(a.evidence.clone());
        })
    }

    /// Appends a re-attestation to "reattestations".
    pub fn reattested(&self, a: &Attested) -> Result<(), String> {
        let (fingerprints, keys, gateway_keys) = verified_state(a);
        self.update(|doc| {
            let entry = json!({
                "fingerprints": fingerprints,
                "keys": keys,
                "gateway_keys": gateway_keys,
                "evidence": a.evidence,
            });
            match doc.get_mut("reattestations").and_then(Value::as_array_mut) {
                Some(list) => list.push(entry),
                None => doc["reattestations"] = json!([entry]),
            }
        })
    }

    /// "ironwire" or "direct": which process holds the key and reaches NEAR AI.
    pub fn set_inference(&self, mode: &str) -> Result<(), String> {
        self.update(|doc| doc["inference"] = mode.into())
    }

    /// "refused", "pin-failed", "model-refused", "signature-failed",
    /// "unattested-provider". Best effort: the caller is already failing.
    pub fn set_status(&self, status: &str, reason: &str) {
        let written = self.update(|doc| {
            doc["status"] = status.into();
            doc["reason"] = reason.into();
        });
        if let Err(e) = written {
            eprintln!("[rts] cannot record attestation status {status:?}: {e}");
        }
    }

    /// Applies `change` and rewrites the file: a 0600 temp file in the same
    /// directory, synced, then renamed over attestation.json.
    fn update(&self, change: impl FnOnce(&mut Value)) -> Result<(), String> {
        let mut doc = self.doc.lock().unwrap_or_else(|e| e.into_inner());
        change(&mut doc);
        let bytes =
            serde_json::to_vec_pretty(&*doc).map_err(|e| format!("attestation record: {e}"))?;
        let dir = self.session_dir();
        let tmp = dir.join(format!(
            ".{FILE_NAME}.{}.tmp",
            crate::protocol::random_hex(6)
        ));
        let written = (|| {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
            fs::rename(&tmp, &self.path)
        })();
        if let Err(e) = written {
            let _ = fs::remove_file(&tmp);
            return Err(format!("cannot write {}: {e}", self.path.display()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attest::{AttestOptions, Attested, ComposePolicy};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    const DUMMY_KEY: &str = "rts-dummy-key-never-send-to-a-provider";

    fn read(r: &AttestationRecord) -> serde_json::Value {
        serde_json::from_slice(&fs::read(r.path()).unwrap()).unwrap()
    }

    fn all_files(dir: &Path) -> Vec<PathBuf> {
        let mut out = vec![];
        for e in fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                out.extend(all_files(&p));
            } else {
                out.push(p);
            }
        }
        out
    }

    #[test]
    fn the_record_is_private_atomic_and_keeps_history() {
        let d = crate::testutil::tempdir("record");
        let sessions = d.join("sessions");
        let options = AttestOptions {
            allow_tcb_out_of_date: true,
            compose_policy: ComposePolicy::MrConfigOnly,
            allow_unattested_provider: false,
        };
        let r = AttestationRecord::create(&sessions, "0123456789ab", &options).unwrap();
        assert_eq!(r.session_dir(), sessions.join("0123456789ab"));
        assert_eq!(r.path(), sessions.join("0123456789ab/attestation.json"));
        let mode = |p: &Path| fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode(r.session_dir()), 0o700);
        assert_eq!(mode(r.path()), 0o600);
        let doc = read(&r);
        assert_eq!(doc["session"], "0123456789ab");
        assert_eq!(doc["status"], "pending");
        assert_eq!(doc["overrides"]["allow_tcb_out_of_date"], true);
        assert_eq!(doc["overrides"]["compose_policy"], "mr-config-only");

        let mut a = Attested::default();
        a.fingerprints.insert([0xaa; 32]);
        a.keys.entry("m".into()).or_default().insert([0xbb; 32]);
        a.gateway_keys.insert([0xcc; 32]);
        a.evidence.push(serde_json::json!({"model": "m"}));
        r.verified(&a).unwrap();
        r.set_inference("direct").unwrap();
        let doc = read(&r);
        assert_eq!(doc["status"], "verified");
        assert_eq!(doc["inference"], "direct");
        assert_eq!(
            doc["fingerprints"],
            serde_json::json!([hex::encode([0xaa; 32])])
        );
        assert_eq!(
            doc["keys"]["m"],
            serde_json::json!([hex::encode([0xbb; 32])])
        );
        assert_eq!(
            doc["gateway_keys"],
            serde_json::json!([hex::encode([0xcc; 32])])
        );
        assert_eq!(doc["evidence"][0]["model"], "m");

        r.reattested(&a).unwrap();
        r.reattested(&a).unwrap();
        let doc = read(&r);
        let re = doc["reattestations"].as_array().unwrap();
        assert_eq!(re.len(), 2);
        assert_eq!(
            re[1]["gateway_keys"],
            serde_json::json!([hex::encode([0xcc; 32])])
        );

        r.set_status("refused", "x");
        let doc = read(&r);
        assert_eq!(doc["status"], "refused");
        assert_eq!(doc["reason"], "x");
        assert_eq!(
            doc["fingerprints"],
            serde_json::json!([hex::encode([0xaa; 32])])
        );
        assert_eq!(doc["session"], "0123456789ab");

        let files = all_files(r.session_dir());
        assert_eq!(
            files,
            vec![r.path().to_path_buf()],
            "stray files: {files:?}"
        );
        for f in files {
            assert!(!fs::read_to_string(&f).unwrap().contains(DUMMY_KEY));
        }
        fs::remove_dir_all(d).unwrap();
    }

    #[test]
    fn a_session_id_that_is_not_a_plain_name_is_refused() {
        let d = crate::testutil::tempdir("record-id");
        for id in ["", "..", "a/b", "../x", "."] {
            assert!(
                AttestationRecord::create(&d, id, &AttestOptions::default()).is_err(),
                "accepted {id:?}"
            );
        }
        fs::remove_dir_all(d).unwrap();
    }
}
