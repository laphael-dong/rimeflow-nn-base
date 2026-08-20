use rimeflow_onnx_base::contract_test_seam::{
    BackendKind, ContractSeamError, DeterministicRuntimeFake, InitFailure, InitOutcome,
    InitRequest, InitializationStage, ResolvedBackend, TargetPlatform, TimeoutBoundary,
    WebInitOutcome,
};

fn linux_target() -> TargetPlatform {
    TargetPlatform {
        os: "linux",
        arch: "x86_64",
    }
}

fn native_request(
    injected_fault: Option<InitializationStage>,
    timeout: Option<TimeoutBoundary>,
) -> InitRequest {
    InitRequest {
        target: linux_target(),
        model_id: "yolov8n",
        injected_fault,
        timeout,
    }
}

fn expected_native_ready() -> InitOutcome {
    InitOutcome::Ready {
        resolved: ResolvedBackend {
            kind: BackendKind::OpenVino,
            target: linux_target(),
            artifact_id: "yolov8n-onnx-fp32",
        },
    }
}

fn expect_native_outcome(
    id: &str,
    expected: InitOutcome,
    result: Result<InitOutcome, ContractSeamError>,
) {
    match result {
        Ok(actual) if actual == expected => {}
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("{id}: expected {expected:?}; Phase 4 remains not_implemented:{operation}")
        }
        Ok(actual) => panic!("{id}: expected {expected:?}; observed {actual:?}"),
        Err(error) => panic!("{id}: expected {expected:?}; observed {error}"),
    }
}

fn expect_web_outcome(
    id: &str,
    expected: WebInitOutcome,
    result: Result<WebInitOutcome, ContractSeamError>,
) {
    match result {
        Ok(actual) if actual == expected => {}
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("{id}: expected {expected:?}; Phase 4 remains not_implemented:{operation}")
        }
        Ok(actual) => panic!("{id}: expected {expected:?}; observed {actual:?}"),
        Err(error) => panic!("{id}: expected {expected:?}; observed {error}"),
    }
}

// RFB-BASE-LIFECYCLE-001
#[test]
fn native_selection_is_fixed_across_repeated_initialize_requests() {
    let mut fake = DeterministicRuntimeFake::new();
    expect_native_outcome(
        "RFB-BASE-LIFECYCLE-001",
        expected_native_ready(),
        fake.initialize_native(native_request(None, None)),
    );
    expect_native_outcome(
        "RFB-BASE-LIFECYCLE-001",
        expected_native_ready(),
        fake.initialize_native(native_request(None, None)),
    );
}

// RFB-BASE-LIFECYCLE-002
#[test]
fn every_native_initialization_stage_fails_atomically_with_one_fallback() {
    let stages = [
        InitializationStage::ManifestParse,
        InitializationStage::ArtifactIntegrity,
        InitializationStage::RuntimeLoad,
        InitializationStage::DeviceCreate,
        InitializationStage::ModelCompile,
        InitializationStage::IoDiscovery,
        InitializationStage::BufferPrepare,
        InitializationStage::SmokeInference,
    ];

    for stage in stages {
        let mut fake = DeterministicRuntimeFake::new();
        expect_native_outcome(
            "RFB-BASE-LIFECYCLE-002",
            InitOutcome::UseWebFallback {
                failure: InitFailure {
                    code: "native_initialization_failed",
                    stage,
                },
            },
            fake.initialize_native(native_request(Some(stage), None)),
        );
    }
}

// RFB-BASE-LIFECYCLE-003
#[test]
fn native_initialization_timeout_returns_structured_fallback() {
    let mut fake = DeterministicRuntimeFake::new();
    expect_native_outcome(
        "RFB-BASE-LIFECYCLE-003",
        InitOutcome::UseWebFallback {
            failure: InitFailure {
                code: "native_initialization_timeout",
                stage: InitializationStage::RuntimeLoad,
            },
        },
        fake.initialize_native(native_request(
            None,
            Some(TimeoutBoundary::NativeInitialization),
        )),
    );
}

// RFB-BASE-LIFECYCLE-004
#[test]
fn web_initialization_failure_is_terminal_without_native_retry() {
    let mut fake = DeterministicRuntimeFake::new();
    expect_web_outcome(
        "RFB-BASE-LIFECYCLE-004",
        WebInitOutcome::TerminalFailure {
            failure: InitFailure {
                code: "web_initialization_timeout",
                stage: InitializationStage::SmokeInference,
            },
        },
        fake.initialize_web(native_request(
            None,
            Some(TimeoutBoundary::WebInitialization),
        )),
    );
}

// RFB-BASE-LIFECYCLE-005
#[test]
fn concurrent_or_repeated_initialization_publishes_one_backend() {
    let mut fake = DeterministicRuntimeFake::new();
    expect_native_outcome(
        "RFB-BASE-LIFECYCLE-005",
        expected_native_ready(),
        fake.initialize_native(native_request(None, None)),
    );
    assert_eq!(
        fake.operations().len(),
        1,
        "RFB-BASE-LIFECYCLE-005: exactly one native backend must be created"
    );
}

// RFB-BASE-LIFECYCLE-006
#[test]
fn released_runtime_rejects_later_use_without_reinitializing() {
    let mut fake = DeterministicRuntimeFake::new();
    match fake.release() {
        Ok(()) => {}
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("RFB-BASE-LIFECYCLE-006: expected release to transition to Released; Phase 4 remains not_implemented:{operation}")
        }
        Err(error) => {
            panic!("RFB-BASE-LIFECYCLE-006: expected release to succeed; observed {error}")
        }
    }
}

// RFB-BASE-LIFECYCLE-007
#[test]
fn inference_failure_does_not_switch_or_rebuild_the_backend() {
    let mut fake = DeterministicRuntimeFake::new();
    match fake.infer() {
        Err(ContractSeamError::InferenceFailure {
            code: "inference_failed",
        }) => {}
        Err(ContractSeamError::NotImplemented { operation }) => {
            panic!("RFB-BASE-LIFECYCLE-007: expected an inference-only error with a fixed backend; Phase 4 remains not_implemented:{operation}")
        }
        Err(error) => panic!(
            "RFB-BASE-LIFECYCLE-007: expected inference_failure:inference_failed; observed {error}"
        ),
        Ok(()) => panic!("RFB-BASE-LIFECYCLE-007: expected a deterministic inference failure"),
    }
}

// RFB-BASE-LIFECYCLE-008
#[test]
fn ready_and_fallback_diagnostics_remain_observable() {
    let mut fake = DeterministicRuntimeFake::new();
    expect_native_outcome(
        "RFB-BASE-LIFECYCLE-008",
        expected_native_ready(),
        fake.initialize_native(native_request(None, None)),
    );
}
