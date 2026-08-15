#![cfg(all(target_os = "linux", target_arch = "x86_64", feature = "native"))]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use rimeflow_onnx_base::manifest::sha256_hex;
use rimeflow_onnx_base::{
    AdapterConformanceCase, AdapterConformanceCheck, AdapterConformanceCheckKind,
    AdapterConformanceReport, AdapterConformanceStatus, AdapterSelection, BackendInitRequest,
    BackendInstance, BackendKind, ConformanceEvidenceKind, ConformanceRunner, DType, ExecutionPlan,
    LegacyOrtMetadata, LinuxOrtBackend, ModelInput, ModelManifest, NativeAdapterCapability,
    NativeOrtBackend, OneShotNativeAdapterFactory, Platform, PlatformAdapterFactory,
    RuntimeLifecycle, SelectedNativeAdapter, TensorData, ADAPTER_CONFORMANCE_SCHEMA_VERSION,
};

const MODEL_SHA256: &str = "9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad";
const MODEL_BYTES: usize = 12_851_098;
const MANIFEST_JSON: &str = include_str!("fixtures/conformance/linux-ort-manifest.json");
const MANIFEST_SHA256: &str = "6f411aedec1550f3306459468dc3b4a0a4bc2558f5233f5f25404f1ac50e9c26";
const OUTPUT_ELEMENTS: usize = 84 * 8400;

#[test]
#[ignore = "requires the locked Validation model via RIMEFLOW_YOLOV8N_MODEL"]
fn real_linux_ort_adapter_conformance_executes_locked_model() {
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
        BackendKind::LinuxOrt,
        request.target.clone(),
        vec![rimeflow_onnx_base::ArtifactFormat::Onnx],
    );
    capability.configured_provider = Some("CPU".to_owned());
    capability.execution_plan = ExecutionPlan::Unknown;
    capability.runtime_version = Some("ort-2.0.0-rc.12".to_owned());
    let selector = PlatformAdapterFactory::new(manifest, vec![capability]);
    let builder_model = Arc::clone(&model_bytes);
    let factory = OneShotNativeAdapterFactory::new(
        selector,
        move |request: &BackendInitRequest, _selected: &SelectedNativeAdapter| {
            let backend = LinuxOrtBackend::from_model_bytes(
                &builder_model,
                640,
                LegacyOrtMetadata {
                    platform: request.target.clone(),
                    model_version: request.model_version.clone(),
                    artifact_id: request.artifact_id.clone(),
                    artifact_sha256: request.artifact_sha256.clone(),
                    output_role: "detections".to_owned(),
                    output_shape: vec![1, 84, 8400],
                    runtime_version: Some("ort-2.0.0-rc.12".to_owned()),
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
        .expect("Linux ORT initialization");
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
        .expect("Linux ORT smoke inference");
    let inference_ms = inference_started.elapsed().as_millis();
    let TensorData::F32(adapter_values) = &output.tensors[0].data else {
        panic!("Linux ORT output must be f32")
    };
    assert_eq!(adapter_values.len(), OUTPUT_ELEMENTS);
    assert!(adapter_values.iter().all(|value| value.is_finite()));

    let mut native = NativeOrtBackend::from_model_bytes(&model_bytes, 640)
        .expect("reference NativeOrtBackend initializes");
    let reference = native
        .infer_from_host_slice(&input)
        .expect("reference Native ORT inference");
    let max_difference = reference
        .iter()
        .zip(adapter_values)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f32, f32::max);
    assert!(max_difference <= 1e-6, "max difference {max_difference}");

    let error = lifecycle
        .infer(ModelInput::Tensor {
            role: "image".to_owned(),
            shape: vec![1],
            dtype: DType::F32,
            bytes: 0.0f32.to_le_bytes().to_vec(),
        })
        .expect_err("post-readiness inference error is returned");
    assert_eq!(error.code, "inference_failed");
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(factory.selector().selection_evaluation_count(), 1);
    assert_eq!(lifecycle.published_instance_count(), 1);
    assert_eq!(lifecycle.web_fallback_count(), 0);

    let diagnostics = lifecycle.diagnostics().expect("ready diagnostics");
    assert_eq!(diagnostics.backend_kind, BackendKind::LinuxOrt);
    assert_eq!(diagnostics.configured_provider.as_deref(), Some("CPU"));
    assert_eq!(diagnostics.accelerator, None);
    assert_eq!(diagnostics.execution_plan, ExecutionPlan::Unknown);

    let selection = factory
        .selector()
        .selected()
        .expect("cached selection")
        .clone();
    assert!(matches!(selection, AdapterSelection::Ready { .. }));
    let report = AdapterConformanceReport {
        schema_version: ADAPTER_CONFORMANCE_SCHEMA_VERSION,
        case: AdapterConformanceCase {
            id: "linux-x86_64-ort-cpu".to_owned(),
            model_id: request.model_id.clone(),
            model_version: request.model_version.clone(),
            target: request.target.clone(),
            adapter: BackendKind::LinuxOrt,
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
                format!("real ORT inference produced {OUTPUT_ELEMENTS} finite values"),
            ),
            passed(
                AdapterConformanceCheckKind::GoldenOutput,
                format!("Linux adapter matched NativeOrtBackend; max difference {max_difference}"),
            ),
            passed(
                AdapterConformanceCheckKind::FaultInjection,
                "artifact/runtime/device/smoke failure stages are covered by the focused suite",
            ),
            passed(
                AdapterConformanceCheckKind::Diagnostics,
                "resolved backend reports LinuxOrt, CPU, no accelerator, unknown execution plan",
            ),
            AdapterConformanceCheck {
                kind: AdapterConformanceCheckKind::Performance,
                status: AdapterConformanceStatus::BuildVerified,
                detail: format!(
                    "smoke timing captured without a formal threshold: init={initialization_ms}ms infer={inference_ms}ms"
                ),
                evidence_path: Some("tests/linux_ort_conformance.rs".to_owned()),
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
        evidence_path: Some("tests/linux_ort_conformance.rs".to_owned()),
    }
}

fn request() -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("linux", "x86_64"),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "yolov8n-onnx-20260707".to_owned(),
        artifact_id: "linux-onnx-fp32".to_owned(),
        artifact_sha256: MODEL_SHA256.to_owned(),
    }
}
