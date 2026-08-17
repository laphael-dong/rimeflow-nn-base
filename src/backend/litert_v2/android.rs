//! Android binding to Google's official LiteRT 2.1.6 Rust package.

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use litert::model::Tensor;
use litert::{
    CompiledModel, ElementType, EnvironmentBuilder, LiteRtHwAccelerator, Model, Options,
    TensorBuffer,
};

use super::{
    LiteRtCompiledRuntime, LiteRtRuntimeError, LiteRtTensorDescriptor, LiteRtV2Backend,
    LiteRtV2BootstrapError, LiteRtV2Diagnostics, VerifiedLiteRtArtifact, LITERT_RUNTIME_VERSION,
};
use crate::backend::{
    BackendFactory, BackendInitRequest, BackendInstance, BackendKind, DType, ExecutionPlan,
    ModelInput, RawModelOutput, ResolvedBackend, RuntimeBackend, TensorData,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::{ModelManifest, TensorSpec};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AndroidLiteRtAccelerator {
    Cpu,
    Gpu,
    Npu,
}

impl AndroidLiteRtAccelerator {
    fn official(self) -> LiteRtHwAccelerator {
        match self {
            Self::Cpu => LiteRtHwAccelerator::Cpu,
            Self::Gpu => LiteRtHwAccelerator::Gpu,
            Self::Npu => LiteRtHwAccelerator::Npu,
        }
    }

    fn diagnostic_name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Npu => "NPU",
        }
    }
}

pub struct AndroidLiteRtV2Backend {
    inner: LiteRtV2Backend<AndroidCompiledModelRuntime>,
}

impl AndroidLiteRtV2Backend {
    pub fn diagnostics(&self) -> &LiteRtV2Diagnostics {
        self.inner.diagnostics()
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        self.inner.resolved_backend()
    }
}

impl RuntimeBackend for AndroidLiteRtV2Backend {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        self.inner.infer(input)
    }
}

pub struct AndroidLiteRtV2Factory {
    manifest: ModelManifest,
    artifact_root: PathBuf,
    accelerator: AndroidLiteRtAccelerator,
    attempted: AtomicBool,
}

impl AndroidLiteRtV2Factory {
    pub fn new(
        manifest: ModelManifest,
        artifact_root: impl Into<PathBuf>,
        accelerator: AndroidLiteRtAccelerator,
    ) -> Self {
        Self {
            manifest,
            artifact_root: artifact_root.into(),
            accelerator,
            attempted: AtomicBool::new(false),
        }
    }

    pub fn build_attempt_count(&self) -> usize {
        usize::from(self.attempted.load(Ordering::SeqCst))
    }
}

impl BackendFactory<AndroidLiteRtV2Backend> for AndroidLiteRtV2Factory {
    fn create(
        &self,
        request: &BackendInitRequest,
    ) -> Result<BackendInstance<AndroidLiteRtV2Backend>, InitFailure> {
        if self
            .attempted
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return Err(InitFailure::new(
                "native_factory_rebuild_forbidden",
                InitializationStage::RuntimeLoad,
                "the LiteRT CompiledModel factory is one-shot",
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                BackendKind::LiteRtV2,
            ));
        }
        if request.target.os != "android" {
            return Err(LiteRtV2BootstrapError::device(format!(
                "LiteRT Android adapter cannot initialize target {}/{}",
                request.target.os, request.target.arch
            ))
            .into_init_failure(request));
        }
        self.manifest
            .validate_semantics()
            .map_err(|error| manifest_failure(request, "manifest_invalid", error.to_string()))?;
        if self.manifest.model.id != request.model_id
            || self.manifest.model.version != request.model_version
        {
            return Err(manifest_failure(
                request,
                "manifest_identity_mismatch",
                "manifest model identity does not match the initialization request",
            ));
        }

        let artifact = self
            .manifest
            .select_artifact(&request.artifact_id, &request.target)
            .map_err(|error| artifact_failure(request, error.to_string()))?;
        let artifact_path = resolve_artifact_path(&self.artifact_root, &artifact.path)
            .map_err(|message| artifact_failure(request, message))?;
        let artifact_bytes = std::fs::read(&artifact_path).map_err(|error| {
            artifact_failure(
                request,
                format!("failed to read {}: {error}", artifact_path.display()),
            )
        })?;
        let verified =
            VerifiedLiteRtArtifact::verify(self.manifest.clone(), request, &artifact_bytes)?;
        let input_specs = selected_specs(
            &verified.manifest().tensors.inputs,
            &verified.artifact().inputs,
        );
        let output_specs = selected_specs(
            &verified.manifest().tensors.outputs,
            &verified.artifact().outputs,
        );

        let started = Instant::now();
        let runtime = AndroidCompiledModelRuntime::start(
            artifact_bytes,
            input_specs,
            output_specs,
            self.accelerator,
        )
        .map_err(|error| error.into_init_failure(request))?;
        let resolved = ResolvedBackend {
            backend_kind: BackendKind::LiteRtV2,
            platform: request.target.clone(),
            configured_provider: Some("LiteRT CompiledModel".to_owned()),
            accelerator: Some(self.accelerator.diagnostic_name().to_owned()),
            execution_plan: ExecutionPlan::Unknown,
            model_version: request.model_version.clone(),
            artifact_id: request.artifact_id.clone(),
            artifact_sha256: request.artifact_sha256.clone(),
            initialization_ms: 0,
            runtime_version: Some(LITERT_RUNTIME_VERSION.to_owned()),
        };
        let mut inner = LiteRtV2Backend::from_verified_artifact(verified, runtime, resolved)?;
        inner.set_initialization_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
        let resolved = inner.resolved_backend().clone();
        Ok(BackendInstance {
            backend: AndroidLiteRtV2Backend { inner },
            resolved,
        })
    }
}

enum WorkerCommand {
    Run {
        inputs: Vec<TensorData>,
        reply: Sender<Result<Vec<TensorData>, LiteRtRuntimeError>>,
    },
    Shutdown,
}

struct AndroidCompiledModelRuntime {
    inputs: Vec<LiteRtTensorDescriptor>,
    outputs: Vec<LiteRtTensorDescriptor>,
    commands: Sender<WorkerCommand>,
    worker: Option<JoinHandle<()>>,
}

impl AndroidCompiledModelRuntime {
    fn start(
        artifact_bytes: Vec<u8>,
        input_specs: Vec<TensorSpec>,
        output_specs: Vec<TensorSpec>,
        accelerator: AndroidLiteRtAccelerator,
    ) -> Result<Self, LiteRtV2BootstrapError> {
        let (commands, command_receiver) = mpsc::channel();
        let (bootstrap_sender, bootstrap_receiver) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("rimeflow-litert-v2".to_owned())
            .spawn(move || {
                if let Err(error) = compiled_model_worker(
                    artifact_bytes,
                    input_specs,
                    output_specs,
                    accelerator,
                    command_receiver,
                    &bootstrap_sender,
                ) {
                    let _ = bootstrap_sender.send(Err(error));
                }
            })
            .map_err(|error| {
                LiteRtV2BootstrapError::runtime(format!(
                    "failed to start LiteRT owner thread: {error}"
                ))
            })?;
        match bootstrap_receiver.recv() {
            Ok(Ok((inputs, outputs))) => Ok(Self {
                inputs,
                outputs,
                commands,
                worker: Some(worker),
            }),
            Ok(Err(error)) => {
                let _ = worker.join();
                Err(error)
            }
            Err(error) => {
                let _ = worker.join();
                Err(LiteRtV2BootstrapError::runtime(format!(
                    "LiteRT owner thread ended during initialization: {error}"
                )))
            }
        }
    }
}

impl LiteRtCompiledRuntime for AndroidCompiledModelRuntime {
    fn input_descriptors(&self) -> &[LiteRtTensorDescriptor] {
        &self.inputs
    }

    fn output_descriptors(&self) -> &[LiteRtTensorDescriptor] {
        &self.outputs
    }

    fn run(&mut self, inputs: &[TensorData]) -> Result<Vec<TensorData>, LiteRtRuntimeError> {
        let (reply, response) = mpsc::channel();
        self.commands
            .send(WorkerCommand::Run {
                inputs: inputs.to_vec(),
                reply,
            })
            .map_err(|error| {
                LiteRtRuntimeError::new(
                    "litert_worker_unavailable",
                    format!("LiteRT owner thread is unavailable: {error}"),
                )
            })?;
        response.recv().map_err(|error| {
            LiteRtRuntimeError::new(
                "litert_worker_unavailable",
                format!("LiteRT owner thread did not return inference: {error}"),
            )
        })?
    }
}

impl Drop for AndroidCompiledModelRuntime {
    fn drop(&mut self) {
        let _ = self.commands.send(WorkerCommand::Shutdown);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

fn compiled_model_worker(
    mut artifact_bytes: Vec<u8>,
    input_specs: Vec<TensorSpec>,
    output_specs: Vec<TensorSpec>,
    accelerator: AndroidLiteRtAccelerator,
    commands: Receiver<WorkerCommand>,
    bootstrap: &SyncSender<
        Result<(Vec<LiteRtTensorDescriptor>, Vec<LiteRtTensorDescriptor>), LiteRtV2BootstrapError>,
    >,
) -> Result<(), LiteRtV2BootstrapError> {
    let environment = EnvironmentBuilder::build_default().map_err(|error| {
        LiteRtV2BootstrapError::runtime(format!("LiteRT environment creation failed: {error}"))
    })?;
    let model =
        Model::create_model_from_buffer(&environment, &mut artifact_bytes).map_err(|error| {
            LiteRtV2BootstrapError::model_compile(format!("LiteRT model loading failed: {error}"))
        })?;
    let options = Options::create_with_accelerator(accelerator.official()).map_err(|error| {
        LiteRtV2BootstrapError::device(format!("LiteRT accelerator setup failed: {error}"))
    })?;
    let compiled = CompiledModel::create(&environment, &model, &options).map_err(|error| {
        LiteRtV2BootstrapError::model_compile(format!(
            "LiteRT CompiledModel creation failed: {error}"
        ))
    })?;
    let (input_descriptors, output_descriptors) = discover_io(&model, &input_specs, &output_specs)?;
    let input_buffers = compiled
        .create_input_tensor_buffers(&environment, &model, 0)
        .map_err(|error| {
            LiteRtV2BootstrapError::buffer_prepare(format!(
                "LiteRT input buffer creation failed: {error}"
            ))
        })?;
    let output_buffers = compiled
        .create_output_tensor_buffers(&environment, &model, 0)
        .map_err(|error| {
            LiteRtV2BootstrapError::buffer_prepare(format!(
                "LiteRT output buffer creation failed: {error}"
            ))
        })?;
    bootstrap
        .send(Ok((input_descriptors.clone(), output_descriptors.clone())))
        .map_err(|error| {
            LiteRtV2BootstrapError::runtime(format!(
                "failed to publish LiteRT initialization: {error}"
            ))
        })?;

    while let Ok(command) = commands.recv() {
        match command {
            WorkerCommand::Run { inputs, reply } => {
                let result = run_compiled_model(
                    &compiled,
                    &input_buffers,
                    &output_buffers,
                    &output_descriptors,
                    &inputs,
                );
                let _ = reply.send(result);
            }
            WorkerCommand::Shutdown => break,
        }
    }
    Ok(())
}

fn discover_io(
    model: &Model,
    input_specs: &[TensorSpec],
    output_specs: &[TensorSpec],
) -> Result<(Vec<LiteRtTensorDescriptor>, Vec<LiteRtTensorDescriptor>), LiteRtV2BootstrapError> {
    let signature = model.signature(0).map_err(|error| {
        LiteRtV2BootstrapError::io_discovery(format!("LiteRT signature discovery failed: {error}"))
    })?;
    let input_names = signature
        .input_names()
        .map_err(|error| LiteRtV2BootstrapError::io_discovery(error.to_string()))?
        .map(|name| {
            name.map(str::to_owned)
                .map_err(|error| LiteRtV2BootstrapError::io_discovery(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_names = signature
        .output_names()
        .map_err(|error| LiteRtV2BootstrapError::io_discovery(error.to_string()))?
        .map(|name| {
            name.map(str::to_owned)
                .map_err(|error| LiteRtV2BootstrapError::io_discovery(error.to_string()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_signature_bindings(&input_names, input_specs, "input")?;
    validate_signature_bindings(&output_names, output_specs, "output")?;

    let inputs = input_names
        .iter()
        .enumerate()
        .map(|(index, binding_name)| {
            let tensor = signature.input_tensor(index).map_err(|error| {
                LiteRtV2BootstrapError::io_discovery(format!(
                    "LiteRT input binding {binding_name} at signature index {index} discovery failed: {error}"
                ))
            })?;
            descriptor(index, binding_name, &tensor, input_specs, "input")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let outputs = output_names
        .iter()
        .enumerate()
        .map(|(index, binding_name)| {
            let tensor = signature.output_tensor(index).map_err(|error| {
                LiteRtV2BootstrapError::io_discovery(format!(
                    "LiteRT output binding {binding_name} at signature index {index} discovery failed: {error}"
                ))
            })?;
            descriptor(index, binding_name, &tensor, output_specs, "output")
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((inputs, outputs))
}

fn validate_signature_bindings(
    binding_names: &[String],
    specs: &[TensorSpec],
    kind: &str,
) -> Result<(), LiteRtV2BootstrapError> {
    if binding_names.len() != specs.len() {
        return Err(LiteRtV2BootstrapError::io_discovery(format!(
            "LiteRT signature exposes {} {kind}(s), manifest declares {}",
            binding_names.len(),
            specs.len()
        )));
    }
    let mut unique_names = std::collections::HashSet::new();
    if binding_names
        .iter()
        .any(|name| name.is_empty() || !unique_names.insert(name.as_str()))
    {
        return Err(LiteRtV2BootstrapError::io_discovery(format!(
            "LiteRT signature exposes an empty or duplicate {kind} binding name"
        )));
    }
    Ok(())
}

fn descriptor(
    index: usize,
    binding_name: &str,
    tensor: &Tensor<'_>,
    specs: &[TensorSpec],
    kind: &str,
) -> Result<LiteRtTensorDescriptor, LiteRtV2BootstrapError> {
    let name = tensor
        .name()
        .map_err(|error| LiteRtV2BootstrapError::io_discovery(error.to_string()))?;
    let dtype = match tensor
        .element_type()
        .map_err(|error| LiteRtV2BootstrapError::io_discovery(error.to_string()))?
    {
        ElementType::Float32 => DType::F32,
        ElementType::UInt8 => DType::U8,
        ElementType::Int8 => DType::I8,
        unsupported => {
            return Err(LiteRtV2BootstrapError::io_discovery(format!(
                "LiteRT tensor {name} has unsupported element type {unsupported:?}"
            )))
        }
    };
    let ranked_type = tensor.ranked_tensor_type().map_err(|error| {
        LiteRtV2BootstrapError::io_discovery(format!(
            "LiteRT {kind} tensor {name} shape discovery failed: {error}"
        ))
    })?;
    let rank = ranked_type.layout.rank() as usize;
    if rank > ranked_type.layout.dimensions.len() {
        return Err(LiteRtV2BootstrapError::io_discovery(format!(
            "LiteRT {kind} tensor {name} rank {rank} exceeds the binding layout capacity"
        )));
    }
    let shape = ranked_type.layout.dimensions[..rank]
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                LiteRtV2BootstrapError::io_discovery(format!(
                    "LiteRT {kind} tensor {name} has dynamic or invalid shape"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut candidates = specs.iter().filter(|spec| {
        spec.name
            .as_deref()
            .is_none_or(|candidate| candidate == name)
            && spec.index.is_none_or(|candidate| candidate == index)
    });
    let spec = candidates
        .next()
        .ok_or_else(|| {
            LiteRtV2BootstrapError::io_discovery(format!(
                "LiteRT {kind} binding {binding_name} resolves to tensor {name} at index {index}, which has no logical manifest role"
            ))
        })?;
    if candidates.next().is_some() {
        return Err(LiteRtV2BootstrapError::io_discovery(format!(
            "LiteRT {kind} binding {binding_name} resolves ambiguously to tensor {name} at index {index}"
        )));
    }
    let expected_shape = spec
        .shape
        .iter()
        .map(|dimension| {
            usize::try_from(*dimension).map_err(|_| {
                LiteRtV2BootstrapError::io_discovery(format!(
                    "LiteRT {kind} tensor {name} has an invalid manifest shape"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape != expected_shape || dtype != spec.dtype {
        return Err(LiteRtV2BootstrapError::io_discovery(format!(
            "LiteRT {kind} binding {binding_name} resolves to tensor {name} with {shape:?}/{dtype:?}, manifest requires {expected_shape:?}/{:?}",
            spec.dtype
        )));
    }
    Ok(LiteRtTensorDescriptor {
        name: name.to_owned(),
        signature_binding_name: Some(binding_name.to_owned()),
        index,
        shape,
        dtype,
    })
}

fn run_compiled_model(
    compiled: &CompiledModel,
    input_buffers: &[TensorBuffer<'_>],
    output_buffers: &[TensorBuffer<'_>],
    output_descriptors: &[LiteRtTensorDescriptor],
    inputs: &[TensorData],
) -> Result<Vec<TensorData>, LiteRtRuntimeError> {
    if inputs.len() != input_buffers.len() {
        return Err(LiteRtRuntimeError::new(
            "litert_input_count_mismatch",
            format!(
                "expected {} inputs, got {}",
                input_buffers.len(),
                inputs.len()
            ),
        ));
    }
    for (buffer, data) in input_buffers.iter().zip(inputs) {
        write_tensor(buffer, data)?;
    }
    compiled
        .run(0, input_buffers, output_buffers)
        .map_err(|error| LiteRtRuntimeError::new("litert_run_failed", error.to_string()))?;
    output_buffers
        .iter()
        .zip(output_descriptors)
        .map(|(buffer, descriptor)| read_tensor(buffer, descriptor))
        .collect()
}

fn write_tensor(buffer: &TensorBuffer<'_>, data: &TensorData) -> Result<(), LiteRtRuntimeError> {
    let result = match (buffer.element_type(), data) {
        (ElementType::Float32, TensorData::F32(values)) => buffer.write(values),
        (ElementType::Int8, TensorData::I8(values)) => buffer.write(values),
        // The official 0.1.3 binding represents both byte tensor variants via
        // its i8-compatible buffer path; the bit pattern remains unchanged.
        (ElementType::UInt8, TensorData::U8(values)) => {
            let bits = values.iter().map(|value| *value as i8).collect::<Vec<_>>();
            buffer.write(&bits)
        }
        (element_type, _) => {
            return Err(LiteRtRuntimeError::new(
                "litert_input_dtype_mismatch",
                format!("LiteRT buffer expects {element_type:?}"),
            ))
        }
    };
    result
        .map(|_| ())
        .map_err(|error| LiteRtRuntimeError::new("litert_input_write_failed", error.to_string()))
}

fn read_tensor(
    buffer: &TensorBuffer<'_>,
    descriptor: &LiteRtTensorDescriptor,
) -> Result<TensorData, LiteRtRuntimeError> {
    let elements = descriptor
        .shape
        .iter()
        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or_else(|| {
            LiteRtRuntimeError::new("litert_output_shape_overflow", "output shape overflows")
        })?;
    match buffer.element_type() {
        ElementType::Float32 => {
            let mut values = vec![0.0f32; elements];
            buffer.read(&mut values).map_err(|error| {
                LiteRtRuntimeError::new("litert_output_read_failed", error.to_string())
            })?;
            Ok(TensorData::F32(values))
        }
        ElementType::Int8 => {
            let mut values = vec![0i8; elements];
            buffer.read(&mut values).map_err(|error| {
                LiteRtRuntimeError::new("litert_output_read_failed", error.to_string())
            })?;
            Ok(TensorData::I8(values))
        }
        ElementType::UInt8 => {
            let mut bits = vec![0i8; elements];
            buffer.read(&mut bits).map_err(|error| {
                LiteRtRuntimeError::new("litert_output_read_failed", error.to_string())
            })?;
            Ok(TensorData::U8(
                bits.into_iter().map(|value| value as u8).collect(),
            ))
        }
        element_type => Err(LiteRtRuntimeError::new(
            "litert_output_dtype_unsupported",
            format!("unsupported LiteRT output type {element_type:?}"),
        )),
    }
}

fn selected_specs(specs: &[TensorSpec], roles: &[String]) -> Vec<TensorSpec> {
    roles
        .iter()
        .filter_map(|role| specs.iter().find(|spec| spec.role == *role).cloned())
        .collect()
}

fn resolve_artifact_path(root: &Path, manifest_path: &str) -> Result<PathBuf, String> {
    let relative = Path::new(manifest_path);
    if relative.is_absolute()
        || relative.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err("LiteRT artifact path must stay below the configured artifact root".to_owned());
    }
    Ok(root.join(relative))
}

fn artifact_failure(request: &BackendInitRequest, message: impl Into<String>) -> InitFailure {
    InitFailure::new(
        "adapter_or_artifact_unavailable",
        InitializationStage::ArtifactIntegrity,
        message,
    )
    .with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::LiteRtV2,
    )
}

fn manifest_failure(
    request: &BackendInitRequest,
    code: &'static str,
    message: impl Into<String>,
) -> InitFailure {
    InitFailure::new(code, InitializationStage::ManifestParse, message).with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::LiteRtV2,
    )
}
