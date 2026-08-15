use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use rimeflow_onnx_base::{
    BackendInitRequest, BackendKind, CapabilityStatus, DType, ExecutionPlan, InitOutcome,
    InitializationStage, LifecycleSnapshot, MindSporeLiteAdapterBuilder, MindSporeLiteAvailability,
    MindSporeLiteBackend, MindSporeLiteBootstrapError, MindSporeLiteLoadedRuntime,
    MindSporeLiteRuntime, MindSporeLiteRuntimeError, MindSporeLiteRuntimeLoader,
    MindSporeLiteTensorDescriptor, ModelInput, ModelManifest, OneShotNativeAdapterFactory,
    Platform, PlatformAdapterFactory, RuntimeLifecycle, SelectedNativeAdapter, TensorData,
    MINDSPORE_LITE_INFERENCE_TIMEOUT_MS, MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS,
    MINDSPORE_LITE_RUNTIME_VERSION,
};

const ARTIFACT_BYTES: &[u8] = b"mindspore-lite-test-artifact";
const ARTIFACT_SHA256: &str = "bfd1221629fcc0a0b8c8df0df3925e85f8cc845d236e41c5e73a0d2da27ab59d";

#[derive(Clone, Copy)]
enum DescriptorMode {
    Exact,
    ExtraInput,
}

struct FakeLoader {
    loads: Arc<AtomicUsize>,
    runs: Arc<AtomicUsize>,
    last_timeout_ms: Arc<AtomicU64>,
    initialization_ms: u64,
    descriptors: DescriptorMode,
    fail_after_smoke: bool,
}

impl MindSporeLiteRuntimeLoader for FakeLoader {
    type Runtime = FakeRuntime;

    fn load(
        &self,
        artifact: &rimeflow_onnx_base::VerifiedMindSporeLiteArtifact,
        artifact_bytes: &[u8],
        timeout: Duration,
    ) -> Result<MindSporeLiteLoadedRuntime<Self::Runtime>, MindSporeLiteBootstrapError> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        assert_eq!(artifact.request(), &request());
        assert_eq!(artifact_bytes, ARTIFACT_BYTES);
        assert_eq!(
            timeout,
            Duration::from_millis(MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS)
        );

        let mut inputs = vec![input_descriptor()];
        if matches!(self.descriptors, DescriptorMode::ExtraInput) {
            inputs.push(MindSporeLiteTensorDescriptor {
                name: "unexpected".to_owned(),
                index: 1,
                shape: vec![1],
                dtype: DType::F32,
            });
        }
        Ok(MindSporeLiteLoadedRuntime {
            runtime: FakeRuntime {
                inputs,
                outputs: vec![output_descriptor()],
                runs: Arc::clone(&self.runs),
                last_timeout_ms: Arc::clone(&self.last_timeout_ms),
                fail_after_smoke: self.fail_after_smoke,
            },
            initialization_ms: self.initialization_ms,
        })
    }
}

struct FakeRuntime {
    inputs: Vec<MindSporeLiteTensorDescriptor>,
    outputs: Vec<MindSporeLiteTensorDescriptor>,
    runs: Arc<AtomicUsize>,
    last_timeout_ms: Arc<AtomicU64>,
    fail_after_smoke: bool,
}

impl MindSporeLiteRuntime for FakeRuntime {
    fn input_descriptors(&self) -> &[MindSporeLiteTensorDescriptor] {
        &self.inputs
    }

    fn output_descriptors(&self) -> &[MindSporeLiteTensorDescriptor] {
        &self.outputs
    }

    fn run(
        &mut self,
        inputs: &[TensorData],
        timeout: Duration,
    ) -> Result<Vec<TensorData>, MindSporeLiteRuntimeError> {
        assert_eq!(inputs.len(), 1);
        self.last_timeout_ms.store(
            u64::try_from(timeout.as_millis()).expect("bounded timeout"),
            Ordering::SeqCst,
        );
        let run = self.runs.fetch_add(1, Ordering::SeqCst);
        if run > 0 && self.fail_after_smoke {
            return Err(MindSporeLiteRuntimeError::new(
                "mindspore_lite_predict_timeout",
                "injected target prediction timeout",
            ));
        }
        Ok(vec![TensorData::F32(vec![0.0; 84 * 8400])])
    }
}

#[test]
fn invalid_identity_or_bytes_never_reach_the_runtime_loader() {
    let loads = Arc::new(AtomicUsize::new(0));
    let invalid_bytes_builder = builder(
        b"tampered-artifact".to_vec(),
        Arc::clone(&loads),
        7,
        DescriptorMode::Exact,
        false,
    );
    let error = match invalid_bytes_builder.build(&request(), &selected()) {
        Ok(_) => panic!("tampered bytes must fail"),
        Err(error) => error,
    };
    assert_eq!(error.stage, InitializationStage::ArtifactIntegrity);
    assert_eq!(loads.load(Ordering::SeqCst), 0);

    let valid_builder = builder(
        ARTIFACT_BYTES.to_vec(),
        Arc::clone(&loads),
        7,
        DescriptorMode::Exact,
        false,
    );
    let mut wrong_model = request();
    wrong_model.model_version = "wrong-version".to_owned();
    let error = match valid_builder.build(&wrong_model, &selected_for(&wrong_model)) {
        Ok(_) => panic!("wrong model identity must fail"),
        Err(error) => error,
    };
    assert_eq!(error.code.as_ref(), "manifest_identity_mismatch");
    assert_eq!(loads.load(Ordering::SeqCst), 0);

    let mut wrong_target = request();
    wrong_target.target = Platform::new("linux", "x86_64");
    let error = match valid_builder.build(&wrong_target, &selected_for(&wrong_target)) {
        Ok(_) => panic!("wrong target must fail"),
        Err(error) => error,
    };
    assert_eq!(error.code.as_ref(), "adapter_or_artifact_unavailable");
    assert_eq!(loads.load(Ordering::SeqCst), 0);
}

#[test]
fn runtime_io_discovery_rejects_unmapped_descriptors() {
    let loads = Arc::new(AtomicUsize::new(0));
    let builder = builder(
        ARTIFACT_BYTES.to_vec(),
        Arc::clone(&loads),
        7,
        DescriptorMode::ExtraInput,
        false,
    );
    let error = match builder.build(&request(), &selected()) {
        Ok(_) => panic!("extra runtime input must fail"),
        Err(error) => error,
    };
    assert_eq!(error.stage, InitializationStage::IoDiscovery);
    assert_eq!(error.code.as_ref(), "mindspore_lite_io_contract_mismatch");
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}

#[test]
fn runtime_load_overrun_is_a_structured_timeout_before_smoke() {
    let loads = Arc::new(AtomicUsize::new(0));
    let builder = builder(
        ARTIFACT_BYTES.to_vec(),
        Arc::clone(&loads),
        MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS + 1,
        DescriptorMode::Exact,
        false,
    );
    let error = match builder.build(&request(), &selected()) {
        Ok(_) => panic!("runtime load overrun must fail"),
        Err(error) => error,
    };
    assert_eq!(error.code.as_ref(), "native_initialization_timeout");
    assert_eq!(error.stage, InitializationStage::RuntimeLoad);
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}

#[test]
fn smoke_uses_the_remaining_native_initialization_budget() {
    let loads = Arc::new(AtomicUsize::new(0));
    let runs = Arc::new(AtomicUsize::new(0));
    let last_timeout_ms = Arc::new(AtomicU64::new(0));
    let builder = MindSporeLiteAdapterBuilder::new(
        manifest(),
        ARTIFACT_BYTES.to_vec(),
        FakeLoader {
            loads: Arc::clone(&loads),
            runs: Arc::clone(&runs),
            last_timeout_ms: Arc::clone(&last_timeout_ms),
            initialization_ms: 7,
            descriptors: DescriptorMode::Exact,
            fail_after_smoke: false,
        },
    );

    let backend = builder
        .build(&request(), &selected())
        .expect("smoke completes within the remaining initialization budget");
    assert_eq!(backend.resolved.initialization_ms, 7);
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(
        last_timeout_ms.load(Ordering::SeqCst),
        MINDSPORE_LITE_NATIVE_INITIALIZATION_TIMEOUT_MS - 7
    );
}

#[test]
fn verification_failure_selects_one_stable_web_fallback() {
    let manifest = manifest();
    let loads = Arc::new(AtomicUsize::new(0));
    let builder = builder(
        b"tampered-artifact".to_vec(),
        Arc::clone(&loads),
        7,
        DescriptorMode::Exact,
        false,
    );
    let factory = OneShotNativeAdapterFactory::<MindSporeLiteBackend<FakeRuntime>, _>::new(
        PlatformAdapterFactory::new(manifest, vec![capability()]),
        move |request: &BackendInitRequest, selected: &SelectedNativeAdapter| {
            builder.build(request, selected)
        },
    );
    let lifecycle: RuntimeLifecycle<MindSporeLiteBackend<FakeRuntime>> = RuntimeLifecycle::new();

    let first = lifecycle
        .initialize_native(&request(), &factory)
        .expect("structured fallback");
    let second = lifecycle
        .initialize_native(&request(), &factory)
        .expect("cached fallback");
    assert_eq!(first, second);
    assert!(matches!(
        first,
        InitOutcome::UseWebFallback { ref failure }
            if failure.stage == InitializationStage::ArtifactIntegrity
    ));
    assert_eq!(factory.selector().selection_evaluation_count(), 1);
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(lifecycle.web_fallback_count(), 1);
    assert_eq!(lifecycle.snapshot(), LifecycleSnapshot::UseWebFallback);
    assert_eq!(loads.load(Ordering::SeqCst), 0);
}

#[test]
fn ready_inference_errors_do_not_rebuild_or_switch_backends() {
    let manifest = manifest();
    let loads = Arc::new(AtomicUsize::new(0));
    let runs = Arc::new(AtomicUsize::new(0));
    let last_timeout_ms = Arc::new(AtomicU64::new(0));
    let builder = MindSporeLiteAdapterBuilder::new(
        manifest.clone(),
        ARTIFACT_BYTES.to_vec(),
        FakeLoader {
            loads: Arc::clone(&loads),
            runs: Arc::clone(&runs),
            last_timeout_ms: Arc::clone(&last_timeout_ms),
            initialization_ms: 7,
            descriptors: DescriptorMode::Exact,
            fail_after_smoke: true,
        },
    );
    let factory = OneShotNativeAdapterFactory::<MindSporeLiteBackend<FakeRuntime>, _>::new(
        PlatformAdapterFactory::new(manifest, vec![capability()]),
        move |request: &BackendInitRequest, selected: &SelectedNativeAdapter| {
            builder.build(request, selected)
        },
    );
    let lifecycle: RuntimeLifecycle<MindSporeLiteBackend<FakeRuntime>> = RuntimeLifecycle::new();

    let first = lifecycle
        .initialize_native(&request(), &factory)
        .expect("MindSpore Lite becomes Ready");
    let second = lifecycle
        .initialize_native(&request(), &factory)
        .expect("Ready selection is cached");
    assert_eq!(first, second);
    assert!(matches!(first, InitOutcome::Ready { .. }));
    for _ in 0..2 {
        let error = lifecycle
            .infer(valid_input())
            .expect_err("target prediction failure is returned");
        assert_eq!(error.code, "mindspore_lite_predict_timeout");
    }

    let diagnostics = lifecycle.diagnostics().expect("Ready diagnostics");
    assert_eq!(diagnostics.backend_kind, BackendKind::MindSporeLite);
    assert_eq!(
        diagnostics.configured_provider.as_deref(),
        Some("MindSpore Lite NDK")
    );
    assert_eq!(
        diagnostics.runtime_version.as_deref(),
        Some(MINDSPORE_LITE_RUNTIME_VERSION)
    );
    assert_eq!(diagnostics.initialization_ms, 7);
    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 3);
    assert_eq!(
        last_timeout_ms.load(Ordering::SeqCst),
        MINDSPORE_LITE_INFERENCE_TIMEOUT_MS
    );
    assert_eq!(factory.selector().selection_evaluation_count(), 1);
    assert_eq!(factory.build_attempt_count(), 1);
    assert_eq!(lifecycle.published_instance_count(), 1);
    assert_eq!(lifecycle.web_fallback_count(), 0);
    assert_eq!(lifecycle.snapshot(), LifecycleSnapshot::ReadyNative);
}

fn builder(
    bytes: Vec<u8>,
    loads: Arc<AtomicUsize>,
    initialization_ms: u64,
    descriptors: DescriptorMode,
    fail_after_smoke: bool,
) -> MindSporeLiteAdapterBuilder<FakeLoader> {
    MindSporeLiteAdapterBuilder::new(
        manifest(),
        bytes,
        FakeLoader {
            loads,
            runs: Arc::new(AtomicUsize::new(0)),
            last_timeout_ms: Arc::new(AtomicU64::new(0)),
            initialization_ms,
            descriptors,
            fail_after_smoke,
        },
    )
}

fn manifest() -> ModelManifest {
    ModelManifest::parse_and_validate(include_str!(
        "fixtures/conformance/mindspore-lite-manifest.json"
    ))
    .expect("valid MindSpore Lite fixture manifest")
}

fn request() -> BackendInitRequest {
    BackendInitRequest {
        target: Platform::new("harmonyos", "arm64-v8a"),
        model_id: "rimeflow-yolov8n".to_owned(),
        model_version: "mindspore-lite-test-v1".to_owned(),
        artifact_id: "harmonyos-mindspore-fp32".to_owned(),
        artifact_sha256: ARTIFACT_SHA256.to_owned(),
    }
}

fn selected() -> SelectedNativeAdapter {
    selected_for(&request())
}

fn selected_for(request: &BackendInitRequest) -> SelectedNativeAdapter {
    SelectedNativeAdapter {
        backend_kind: BackendKind::MindSporeLite,
        platform: request.target.clone(),
        artifact_id: request.artifact_id.clone(),
        artifact_sha256: request.artifact_sha256.clone(),
        configured_provider: Some("MindSpore Lite NDK".to_owned()),
        accelerator: None,
        execution_plan: ExecutionPlan::Unknown,
        runtime_version: Some(MINDSPORE_LITE_RUNTIME_VERSION.to_owned()),
    }
}

fn capability() -> rimeflow_onnx_base::NativeAdapterCapability {
    MindSporeLiteAvailability {
        artifact: CapabilityStatus::Ready,
        runtime: CapabilityStatus::Ready,
        device: CapabilityStatus::Ready,
        smoke: CapabilityStatus::Ready,
        accelerator: None,
    }
    .into_native_capability()
}

fn input_descriptor() -> MindSporeLiteTensorDescriptor {
    MindSporeLiteTensorDescriptor {
        name: "images".to_owned(),
        index: 0,
        shape: vec![1, 640, 640, 3],
        dtype: DType::F32,
    }
}

fn output_descriptor() -> MindSporeLiteTensorDescriptor {
    MindSporeLiteTensorDescriptor {
        name: "output0".to_owned(),
        index: 0,
        shape: vec![1, 84, 8400],
        dtype: DType::F32,
    }
}

fn valid_input() -> ModelInput {
    ModelInput::Tensor {
        role: "image".to_owned(),
        shape: vec![1, 640, 640, 3],
        dtype: DType::F32,
        bytes: vec![0; 640 * 640 * 3 * 4],
    }
}
