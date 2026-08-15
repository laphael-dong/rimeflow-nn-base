use rimeflow_onnx_base::contract_test_seam::{
    AdapterConformanceCase, AdapterConformanceOutcome, BackendKind, ContractSeamError,
    DeterministicRuntimeFake, InitFailure, InitializationStage, ResolvedBackend, TargetPlatform,
};

fn expect_conformance(
    id: &str,
    expected: AdapterConformanceOutcome,
    result: Result<AdapterConformanceOutcome, ContractSeamError>,
) {
    match result {
        Ok(actual) if actual == expected => {}
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("{id}: expected adapter conformance {expected:?}; Phase 4 remains not_implemented:{operation}")
        }
        Ok(actual) => {
            panic!("{id}: expected adapter conformance {expected:?}; observed {actual:?}")
        }
        Err(error) => panic!("{id}: expected adapter conformance {expected:?}; observed {error}"),
    }
}

// RFB-BASE-ADAPTER-001
#[test]
fn supported_platform_adapter_conforms_to_manifest_contract() {
    let target = TargetPlatform {
        os: "macos",
        arch: "aarch64",
    };
    let expected = AdapterConformanceOutcome::Ready {
        resolved: ResolvedBackend {
            kind: BackendKind::CoreMl,
            target: target.clone(),
            artifact_id: "yolov8n-coreml-fp32",
        },
    };
    let mut fake = DeterministicRuntimeFake::new();
    expect_conformance(
        "RFB-BASE-ADAPTER-001",
        expected,
        fake.verify_adapter_conformance(AdapterConformanceCase {
            target,
            adapter: BackendKind::CoreMl,
            artifact_id: "yolov8n-coreml-fp32",
            runtime_evidence_available: true,
        }),
    );
}

// RFB-BASE-ADAPTER-002
#[test]
fn unavailable_adapter_or_artifact_returns_native_fallback_diagnostic() {
    let target = TargetPlatform {
        os: "android",
        arch: "arm64",
    };
    let expected = AdapterConformanceOutcome::UseWebFallback {
        failure: InitFailure {
            code: "adapter_or_artifact_unavailable",
            stage: InitializationStage::ArtifactIntegrity,
        },
    };
    let mut fake = DeterministicRuntimeFake::new();
    expect_conformance(
        "RFB-BASE-ADAPTER-002",
        expected,
        fake.verify_adapter_conformance(AdapterConformanceCase {
            target,
            adapter: BackendKind::LiteRtV2,
            artifact_id: "yolov8n-tflite-u8",
            runtime_evidence_available: false,
        }),
    );
}
