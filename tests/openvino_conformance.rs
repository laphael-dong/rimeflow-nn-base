#![cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    feature = "openvino-runtime"
))]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rimeflow_onnx_base::manifest::sha256_hex;
use rimeflow_onnx_base::{
    AdapterConformanceCase, AdapterConformanceCheck, AdapterConformanceCheckKind,
    AdapterConformanceReport, AdapterConformanceStatus, AdapterSelection, BackendInitRequest,
    BackendInstance, BackendKind, ConformanceEvidenceKind, ConformanceRunner, DType, ExecutionPlan,
    ModelInput, ModelManifest, NativeAdapterCapability, OneShotNativeAdapterFactory,
    OpenVinoBackend, OpenVinoMetadata, Platform, PlatformAdapterFactory, RuntimeLifecycle,
    SelectedNativeAdapter, TensorData, ADAPTER_CONFORMANCE_SCHEMA_VERSION,
};

const MODEL_SHA256: &str = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad";
const MODEL_BYTES: usize = 12_851_098;
const MANIFEST_JSON: &str = include_str!("fixtures/conformance/linux-openvino-manifest.json");
const MANIFEST_SHA256: &str = "602e0b1d63abc8750d2af8534d233f5de61ff9531a8c952ee792668160038a07";
const OUTPUT_ELEMENTS: usize = 84 * 8400;

#[test]
#[ignore = "requires the locked Validation model and an OpenVINO Runtime resource directory"]
fn real_linux_openvino_adapter_conformance_executes_locked_model() {
    let model_path = PathBuf::from(
        std::env::var_os("RIMEFLOW_YOLOV8N_MODEL")
            .expect("RIMEFLOW_YOLOV8N_MODEL must name the locked model"),
    );
    let model_bytes = Arc::new(fs::read(&model_path).expect("read locked model"));
    assert_eq!(model_bytes.len(), MODEL_BYTES);
    assert_eq!(sha256_hex(&model_bytes), MODEL_SHA256);
    assert_eq!(sha256_hex(MANIFEST_JSON.as_bytes()), MANIFEST_SHA256);

    let manifest = ModelManifest::parse_and_validate(MANIFEST_JSON).expect("conformance manifest");
    let request = request();
    let mut capability = NativeAdapterCapability::ready(
        BackendKind::OpenVino,
        request.target.clone(),
        vec![rimeflow_onnx_base::ArtifactFormat::Onnx],
    );
    capability.configured_provider = Some("OpenVINO Runtime".to_owned());
    capability.accelerator = Some("CPU".to_owned());
    capability.execution_plan = ExecutionPlan::Full;
    let selector = PlatformAdapterFactory::new(manifest, vec![capability]);
    let builder_model = Arc::clone(&model_bytes);
    let factory = OneShotNativeAdapterFactory::new(
        selector,
        move |request: &BackendInitRequest, _selected: &SelectedNativeAdapter| {
            let runtime = std::env::var_os("RIMEFLOW_OPENVINO_RUNTIME")
                .expect("RIMEFLOW_OPENVINO_RUNTIME must name libopenvino_c.so or its directory");
            let backend = OpenVinoBackend::from_model_bytes_with_runtime(
                &builder_model,
                runtime,
                OpenVinoMetadata {
                    platform: request.target.clone(),
                    model_version: request.model_version.clone(),
                    artifact_id: request.artifact_id.clone(),
                    artifact_sha256: request.artifact_sha256.clone(),
                    input_role: "image".to_owned(),
                    input_shape: vec![1, 3, 640, 640],
                    output_role: "detections".to_owned(),
                    output_shape: vec![1, 84, 8400],
                },
            )?;
            let resolved = backend.resolved_backend().clone();
            Ok(BackendInstance { backend, resolved })
        },
    );
    let lifecycle = RuntimeLifecycle::new();

    let initialization_started = Instant::now();
    let first = lifecycle
        .initialize_native(&request, &factory)
        .expect("Linux OpenVINO initialization");
    let second = lifecycle
        .initialize_native(&request, &factory)
        .expect("repeated initialization");
    let initialization_ms = initialization_started.elapsed().as_millis();
    assert_eq!(first, second);
    assert!(matches!(
        first,
        rimeflow_onnx_base::InitOutcome::Ready { .. }
    ));

    let input = vec![0.0f32; 3 * 640 * 640];
    let inference_started = Instant::now();
    let output = lifecycle
        .infer(ModelInput::Tensor {
            role: "image".to_owned(),
            shape: vec![1, 3, 640, 640],
            dtype: DType::F32,
            bytes: bytemuck::cast_slice(&input).to_vec(),
        })
        .expect("Linux OpenVINO smoke inference");
    let inference_ms = inference_started.elapsed().as_millis();
    let TensorData::F32(adapter_values) = &output.tensors[0].data else {
        panic!("Linux OpenVINO output must be f32")
    };
    assert_eq!(adapter_values.len(), OUTPUT_ELEMENTS);
    assert!(adapter_values.iter().all(|value| value.is_finite()));

    let repeat = lifecycle
        .infer(ModelInput::Tensor {
            role: "image".to_owned(),
            shape: vec![1, 3, 640, 640],
            dtype: DType::F32,
            bytes: bytemuck::cast_slice(&input).to_vec(),
        })
        .expect("repeat OpenVINO inference");
    let TensorData::F32(repeat_values) = &repeat.tensors[0].data else {
        panic!("repeat Linux OpenVINO output must be f32")
    };
    let max_difference = repeat_values
        .iter()
        .zip(adapter_values)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(max_difference, 0.0, "OpenVINO repeat output changed");

    let error = lifecycle
        .infer(ModelInput::Tensor {
            role: "image".to_owned(),
            shape: vec![1],
            dtype: DType::F32,
            bytes: 0.0f32.to_le_bytes().to_vec(),
        })
        .expect_err("post-readiness inference error is returned");
    assert_eq!(error.code, "input_contract_mismatch");
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(factory.selector().selection_evaluation_count(), 1);
    assert_eq!(lifecycle.published_instance_count(), 1);
    assert_eq!(lifecycle.web_fallback_count(), 0);

    let diagnostics = lifecycle.diagnostics().expect("ready diagnostics");
    assert_eq!(diagnostics.backend_kind, BackendKind::OpenVino);
    assert_eq!(
        diagnostics.configured_provider.as_deref(),
        Some("OpenVINO Runtime")
    );
    assert_eq!(diagnostics.accelerator.as_deref(), Some("CPU"));
    assert_eq!(diagnostics.execution_plan, ExecutionPlan::Full);
    assert!(diagnostics.runtime_version.is_some());

    let selection = factory
        .selector()
        .selected()
        .expect("cached selection")
        .clone();
    assert!(matches!(selection, AdapterSelection::Ready { .. }));
    let report = AdapterConformanceReport {
        schema_version: ADAPTER_CONFORMANCE_SCHEMA_VERSION,
        case: AdapterConformanceCase {
            id: "linux-x86_64-openvino-cpu".to_owned(),
            model_id: request.model_id.clone(),
            model_version: request.model_version.clone(),
            target: request.target.clone(),
            adapter: BackendKind::OpenVino,
            artifact_id: request.artifact_id.clone(),
            artifact_sha256: request.artifact_sha256.clone(),
            manifest_sha256: MANIFEST_SHA256.to_owned(),
            native_initialization_timeout_ms: 15_000,
        },
        runner: ConformanceRunner {
            kind: ConformanceEvidenceKind::RealTarget,
            target: request.target.clone(),
            runner_id: Some("local-linux-x86_64".to_owned()),
        },
        selection,
        checks: vec![
            passed(
                AdapterConformanceCheckKind::ManifestIo,
                "manifest roles, shapes, dtype, target, and artifact digest validated",
            ),
            passed(
                AdapterConformanceCheckKind::InitializationTimeout,
                "structured timeout fallback is covered by the focused factory/lifecycle suite",
            ),
            passed(
                AdapterConformanceCheckKind::SmokeInference,
                format!("direct OpenVINO inference produced {OUTPUT_ELEMENTS} finite values"),
            ),
            AdapterConformanceCheck {
                kind: AdapterConformanceCheckKind::GoldenOutput,
                status: AdapterConformanceStatus::BuildVerified,
                detail: format!(
                    "OpenVINO repeat inference was deterministic; model-level golden comparison remains owned by Validation; max difference {max_difference}"
                ),
                evidence_path: Some("tests/openvino_conformance.rs".to_owned()),
            },
            passed(
                AdapterConformanceCheckKind::FaultInjection,
                "artifact/runtime/device/smoke failure stages are covered by the focused suite",
            ),
            passed(
                AdapterConformanceCheckKind::Diagnostics,
                "resolved backend reports direct OpenVINO Runtime, CPU, and a full execution plan",
            ),
            AdapterConformanceCheck {
                kind: AdapterConformanceCheckKind::Performance,
                status: AdapterConformanceStatus::BuildVerified,
                detail: format!(
                    "smoke timing captured without a formal threshold: init={initialization_ms}ms infer={inference_ms}ms"
                ),
                evidence_path: Some("tests/openvino_conformance.rs".to_owned()),
            },
            AdapterConformanceCheck {
                kind: AdapterConformanceCheckKind::PackageLoad,
                status: AdapterConformanceStatus::Blocked,
                detail: "Base verified direct artifact load; final product package-load is owned by RimeCut and was not claimed"
                    .to_owned(),
                evidence_path: None,
            },
        ],
    };
    report.validate().expect("honest Linux conformance report");
    assert_eq!(report.overall_status(), AdapterConformanceStatus::Blocked);
}

fn passed(kind: AdapterConformanceCheckKind, detail: impl Into<String>) -> AdapterConformanceCheck {
    AdapterConformanceCheck {
        kind,
        status: AdapterConformanceStatus::Passed,
        detail: detail.into(),
        evidence_path: Some("tests/openvino_conformance.rs".to_owned()),
    }
}

fn request() -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("linux", "x86_64"),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-onnx-20260707".to_owned(),
        artifact_id: "linux-openvino-onnx-fp32".to_owned(),
        artifact_sha256: MODEL_SHA256.to_owned(),
    }
}
