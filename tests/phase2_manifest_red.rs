use std::fs;
use std::path::PathBuf;

use rimeflow_onnx_base::contract_test_seam::{ContractSeamError, DeterministicRuntimeFake};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/manifest")
        .join(name);
    let contents = fs::read_to_string(&path).expect("manifest fixture must be present");
    serde_json::from_str::<serde_json::Value>(&contents)
        .unwrap_or_else(|error| panic!("fixture {} must be valid JSON: {error}", path.display()));
    contents
}

fn expect_implemented<T>(id: &str, behavior: &str, result: Result<T, ContractSeamError>) -> T {
    match result {
        Ok(value) => value,
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("{id}: expected {behavior}; Phase 4 remains not_implemented:{operation}")
        }
        Err(error) => panic!("{id}: expected {behavior}; observed {error}"),
    }
}

fn expect_rejection(id: &str, expected_code: &str, result: Result<(), ContractSeamError>) {
    match result {
        Err(ContractSeamError::ManifestRejected { code }) if code == expected_code => {}
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("{id}: expected manifest_rejected:{expected_code}; Phase 4 remains not_implemented:{operation}")
        }
        Err(error) => panic!("{id}: expected manifest_rejected:{expected_code}; observed {error}"),
        Ok(()) => panic!("{id}: expected manifest_rejected:{expected_code}; manifest was accepted"),
    }
}

// RFB-BASE-MANIFEST-001
#[test]
fn valid_manifest_is_accepted_by_schema_and_semantics() {
    let manifest = fixture("valid-yolov8n.json");
    let mut fake = DeterministicRuntimeFake::new();
    expect_implemented(
        "RFB-BASE-MANIFEST-001",
        "a valid v1 manifest to pass schema validation",
        fake.validate_manifest_schema(&manifest),
    );
    expect_implemented(
        "RFB-BASE-MANIFEST-001",
        "the same valid manifest to pass Rust semantic validation",
        fake.validate_manifest_semantics(&manifest),
    );
}

// RFB-BASE-MANIFEST-002
#[test]
fn structural_manifest_error_is_rejected_consistently() {
    let manifest = fixture("invalid-structure.json");
    let mut fake = DeterministicRuntimeFake::new();
    expect_rejection(
        "RFB-BASE-MANIFEST-002",
        "manifest_schema_invalid",
        fake.validate_manifest_schema(&manifest),
    );
}

// RFB-BASE-MANIFEST-003
#[test]
fn unknown_schema_version_is_rejected_before_initialization() {
    let manifest = fixture("unknown-schema-version.json");
    let mut fake = DeterministicRuntimeFake::new();
    expect_rejection(
        "RFB-BASE-MANIFEST-003",
        "manifest_schema_version_unsupported",
        fake.validate_manifest_semantics(&manifest),
    );
}

// RFB-BASE-MANIFEST-004
#[test]
fn artifact_integrity_or_target_mismatch_is_rejected() {
    let manifest = fixture("artifact-integrity-mismatch.json");
    let mut fake = DeterministicRuntimeFake::new();
    expect_rejection(
        "RFB-BASE-MANIFEST-004",
        "artifact_integrity_or_target_mismatch",
        fake.validate_manifest_semantics(&manifest),
    );
}

// RFB-BASE-MANIFEST-005
#[test]
fn quantized_tensor_requires_cross_field_parameters() {
    let manifest = fixture("invalid-quantization.json");
    let mut fake = DeterministicRuntimeFake::new();
    expect_rejection(
        "RFB-BASE-MANIFEST-005",
        "quantization_zero_point_missing",
        fake.validate_manifest_semantics(&manifest),
    );
}
