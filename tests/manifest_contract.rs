use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rimeflow_onnx_base::build_helper::validate_manifest_artifact;
use rimeflow_onnx_base::manifest::{sha256_hex, MODEL_MANIFEST_SCHEMA_V1};
use rimeflow_onnx_base::{DType, Layout, ModelManifest, Platform};

fn fixture(name: &str) -> String {
    fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/manifest")
            .join(name),
    )
    .expect("fixture must be readable")
}

#[test]
fn fixed_fixture_schema_and_rust_conclusions_agree() {
    let cases = [
        ("valid-yolov8n.json", true),
        ("invalid-structure.json", false),
        ("unknown-schema-version.json", false),
        ("artifact-integrity-mismatch.json", false),
        ("invalid-quantization.json", false),
    ];

    for (name, expected) in cases {
        let json = fixture(name);
        let schema_accepts = ModelManifest::validate_schema_json(&json).is_ok();
        let rust_accepts = ModelManifest::parse_and_validate(&json).is_ok();
        assert_eq!(schema_accepts, expected, "schema conclusion for {name}");
        assert_eq!(rust_accepts, expected, "Rust conclusion for {name}");
        assert_eq!(schema_accepts, rust_accepts, "validator parity for {name}");
    }
}

#[test]
fn parsed_manifest_preserves_static_tensor_and_role_contract() {
    let manifest =
        ModelManifest::parse_and_validate(&fixture("valid-yolov8n.json")).expect("valid manifest");
    let input = &manifest.tensors.inputs[0];
    assert_eq!(input.role, "image");
    assert_eq!(input.name.as_deref(), Some("images"));
    assert_eq!(input.shape, [1, 3, 640, 640]);
    assert_eq!(input.layout, Layout::Nchw);
    assert_eq!(input.dtype, DType::F32);
    assert_eq!(manifest.artifacts[0].inputs, ["image"]);
    assert_eq!(manifest.artifacts[0].outputs, ["detections"]);
}

#[test]
fn semantic_validation_rejects_static_shape_and_role_drift() {
    let mut value: serde_json::Value =
        serde_json::from_str(&fixture("valid-yolov8n.json")).expect("fixture JSON");
    value["tensors"]["inputs"][0]["shape"][2] = serde_json::json!(0);
    let shape_error = ModelManifest::parse_and_validate(&value.to_string())
        .expect_err("zero static dimension must fail");
    assert_eq!(shape_error.code(), "static_shape_invalid");

    let mut value: serde_json::Value =
        serde_json::from_str(&fixture("valid-yolov8n.json")).expect("fixture JSON");
    value["artifacts"][0]["outputs"][0] = serde_json::json!("missing-role");
    let role_error = ModelManifest::parse_and_validate(&value.to_string())
        .expect_err("unknown output role must fail");
    assert_eq!(role_error.code(), "manifest_role_invalid");
}

#[test]
fn build_helper_selects_target_and_verifies_exact_artifact_hash() {
    let root = unique_temp_dir("manifest-build-helper");
    fs::create_dir_all(&root).expect("create test directory");
    let artifact_bytes = b"locked model fixture";
    let artifact_path = root.join("model.onnx");
    fs::write(&artifact_path, artifact_bytes).expect("write artifact");
    let target = Platform::current();
    let manifest_path = root.join("manifest.json");
    let manifest = serde_json::json!({
        "schemaVersion": 1,
        "model": { "id": "fixture", "version": "1.0.0" },
        "tensors": {
            "inputs": [{
                "role": "image", "name": "images", "shape": [1, 3, 1, 1],
                "layout": "NCHW", "dtype": "f32"
            }],
            "outputs": [{
                "role": "detections", "name": "output0", "shape": [1, 1],
                "layout": "NCHW", "dtype": "f32"
            }]
        },
        "artifacts": [{
            "id": "fixture-onnx", "format": "onnx",
            "targets": [{ "os": target.os, "arch": target.arch }],
            "path": "model.onnx", "sha256": sha256_hex(artifact_bytes),
            "converter": { "name": "source", "version": "1" },
            "inputs": ["image"], "outputs": ["detections"]
        }]
    });
    fs::write(&manifest_path, manifest.to_string()).expect("write manifest");

    let selected = validate_manifest_artifact(&manifest_path, "fixture-onnx", &target)
        .expect("build validation succeeds");
    assert_eq!(selected.artifact_path, artifact_path);
    assert_eq!(selected.artifact.sha256, sha256_hex(artifact_bytes));

    let wrong_target = Platform::new("unsupported-os", "unsupported-arch");
    let error = validate_manifest_artifact(&manifest_path, "fixture-onnx", &wrong_target)
        .expect_err("target mismatch must fail");
    assert!(error.to_string().contains("target mismatch"));

    fs::write(&artifact_path, b"mutated model fixture").expect("mutate artifact");
    let error = validate_manifest_artifact(&manifest_path, "fixture-onnx", &target)
        .expect_err("hash mismatch must fail");
    assert!(error.to_string().contains("expected"));

    fs::remove_dir_all(&root).expect("remove isolated test directory");
}

#[test]
fn checked_in_schema_is_valid_json_and_declares_cross_field_guards() {
    let schema: serde_json::Value =
        serde_json::from_str(MODEL_MANIFEST_SCHEMA_V1).expect("schema JSON");
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(schema["properties"]["schemaVersion"]["const"], 1);
    assert!(schema["$defs"]["tensors"]["items"]["allOf"].is_array());
    assert!(
        schema["$defs"]["artifact"]["properties"]["sha256"]["pattern"]
            .as_str()
            .expect("hash pattern")
            .contains("?!0{64}")
    );
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "rimeflow-onnx-base-{label}-{}-{timestamp}",
        std::process::id()
    ))
}
