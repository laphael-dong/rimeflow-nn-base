#![cfg(all(
    feature = "native-coreml-adapter",
    any(target_os = "macos", target_os = "ios")
))]

use std::path::PathBuf;

use rimeflow_onnx_base::{
    BackendInitRequest, CoreMlBackend, ModelInput, ModelManifest, Platform, RuntimeBackend,
    TensorData, DEFAULT_COREML_INITIALIZATION_TIMEOUT,
};

const MANIFEST_JSON: &str = include_str!("fixtures/conformance/coreml-manifest.json");
const ARTIFACT_SHA256: &str = "299e6218590fb62da49407e334431a43a999a96020bdf52b4ebc04708218fb98";

#[test]
#[ignore = "requires the locked Validation .mlpackage on a real macOS/iOS runner"]
fn real_coreml_package_load_and_smoke_inference() {
    let package_path = PathBuf::from(
        std::env::var_os("RIMEFLOW_COREML_PACKAGE")
            .expect("RIMEFLOW_COREML_PACKAGE must name the locked Validation .mlpackage"),
    );
    let target_os = std::env::consts::OS;
    let target_arch = std::env::consts::ARCH;
    let manifest = ModelManifest::parse_and_validate(MANIFEST_JSON).expect("Core ML manifest");
    let request = BackendInitRequest {
        target: Platform::new(target_os, target_arch),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-coreml-20260811".to_owned(),
        artifact_id: "apple-coreml-fp32".to_owned(),
        artifact_sha256: ARTIFACT_SHA256.to_owned(),
    };
    let mut backend = CoreMlBackend::load_package(
        &package_path,
        &manifest,
        &request,
        DEFAULT_COREML_INITIALIZATION_TIMEOUT,
    )
    .expect("official Core ML package compilation and load");

    let output = backend
        .infer(ModelInput::Rgba8 {
            width: 640,
            height: 640,
            bytes: vec![0; 640 * 640 * 4],
        })
        .expect("Core ML smoke inference");
    assert_eq!(output.tensors.len(), 1);
    assert_eq!(output.tensors[0].role, "detections");
    assert_eq!(output.tensors[0].shape, [1, 84, 8400]);
    let TensorData::F32(values) = &output.tensors[0].data else {
        panic!("Core ML output must be FLOAT32")
    };
    assert_eq!(values.len(), 84 * 8400);
    assert!(values.iter().all(|value| value.is_finite()));

    let diagnostics = backend.resolved_backend();
    assert_eq!(diagnostics.configured_provider.as_deref(), Some("CoreML"));
    assert_eq!(diagnostics.accelerator, None);
    assert_eq!(diagnostics.artifact_sha256, ARTIFACT_SHA256);
}
