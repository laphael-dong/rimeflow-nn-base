//! Direct, target-gated adapter for Apple's Core ML framework.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde::Serialize;

use super::{
    BackendInitRequest, BackendKind, DType, ExecutionPlan, ModelInput, RawModelOutput,
    ResolvedBackend, RuntimeBackend,
};
use crate::error::{InferenceError, InitFailure, InitializationStage};
use crate::manifest::{sha256_hex, ArtifactFormat, Layout, ModelManifest};

pub const DEFAULT_COREML_INITIALIZATION_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMlIoMapping {
    pub input_role: String,
    pub input_feature_name: String,
    pub input_width: u32,
    pub input_height: u32,
    pub output_role: String,
    pub output_feature_name: String,
    pub output_shape: Vec<usize>,
}

impl CoreMlIoMapping {
    pub fn from_manifest(
        manifest: &ModelManifest,
        request: &BackendInitRequest,
    ) -> Result<Self, InitFailure> {
        if manifest.model.id != request.model_id || manifest.model.version != request.model_version
        {
            return Err(manifest_failure(
                "Core ML manifest model identity does not match the initialization request",
                request,
            ));
        }
        let artifact = manifest
            .select_artifact(&request.artifact_id, &request.target)
            .map_err(|error| manifest_failure(error.to_string(), request))?;
        if artifact.format != ArtifactFormat::Coreml || artifact.sha256 != request.artifact_sha256 {
            return Err(manifest_failure(
                "the selected artifact is not the requested Core ML package",
                request,
            ));
        }
        if artifact.inputs.len() != 1 || artifact.outputs.len() != 1 {
            return Err(io_failure(
                "the initial Core ML adapter requires exactly one input and one output role",
                request,
            ));
        }

        let input_role = &artifact.inputs[0];
        let output_role = &artifact.outputs[0];
        let input = manifest
            .tensors
            .inputs
            .iter()
            .find(|tensor| tensor.role == *input_role)
            .ok_or_else(|| {
                io_failure("Core ML input role is absent from tensors.inputs", request)
            })?;
        let output = manifest
            .tensors
            .outputs
            .iter()
            .find(|tensor| tensor.role == *output_role)
            .ok_or_else(|| {
                io_failure(
                    "Core ML output role is absent from tensors.outputs",
                    request,
                )
            })?;
        if input.layout != Layout::Nchw
            || input.dtype != DType::F32
            || input.shape.len() != 4
            || input.shape[0] != 1
            || input.shape[1] != 3
        {
            return Err(io_failure(
                "the Core ML image feature requires static NCHW FLOAT32 [1,3,H,W] input",
                request,
            ));
        }
        if output.dtype != DType::F32 {
            return Err(io_failure(
                "the initial Core ML adapter requires a FLOAT32 output",
                request,
            ));
        }
        let input_height = u32::try_from(input.shape[2])
            .map_err(|_| io_failure("Core ML input height is not a positive u32 value", request))?;
        let input_width = u32::try_from(input.shape[3])
            .map_err(|_| io_failure("Core ML input width is not a positive u32 value", request))?;
        let output_shape = output
            .shape
            .iter()
            .map(|dimension| {
                usize::try_from(*dimension).map_err(|_| {
                    io_failure("Core ML output shape is not positive and static", request)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            input_role: input.role.clone(),
            input_feature_name: input.name.clone().ok_or_else(|| {
                io_failure(
                    "Core ML input role requires a runtime feature name",
                    request,
                )
            })?,
            input_width,
            input_height,
            output_role: output.role.clone(),
            output_feature_name: output.name.clone().ok_or_else(|| {
                io_failure(
                    "Core ML output role requires a runtime feature name",
                    request,
                )
            })?,
            output_shape,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CoreMlPackageIdentity {
    pub tree_sha256: String,
    pub file_count: usize,
    pub total_file_bytes: u64,
}

#[derive(Debug, Serialize)]
struct CanonicalPackageFile {
    bytes: u64,
    path: String,
    sha256: String,
}

pub fn coreml_package_tree_sha256(
    package_path: impl AsRef<Path>,
) -> Result<CoreMlPackageIdentity, io::Error> {
    let package_path = package_path.as_ref();
    if !package_path.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Core ML package path is not a directory",
        ));
    }

    let mut paths = Vec::new();
    collect_package_files(package_path, package_path, &mut paths)?;
    paths.sort();
    let mut files = Vec::with_capacity(paths.len());
    for (relative, absolute) in paths {
        let bytes = fs::read(&absolute)?;
        files.push(CanonicalPackageFile {
            bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
            path: relative,
            sha256: sha256_hex(&bytes),
        });
    }
    let canonical = serde_json::to_vec(&files).map_err(io::Error::other)?;
    let total_file_bytes = files.iter().map(|file| file.bytes).sum();
    Ok(CoreMlPackageIdentity {
        tree_sha256: sha256_hex(&canonical),
        file_count: files.len(),
        total_file_bytes,
    })
}

fn collect_package_files(
    root: &Path,
    directory: &Path,
    files: &mut Vec<(String, PathBuf)>,
) -> Result<(), io::Error> {
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Core ML package trees must not contain symbolic links",
            ));
        }
        if file_type.is_dir() {
            collect_package_files(root, &path, files)?;
        } else if file_type.is_file() {
            let relative = path
                .strip_prefix(root)
                .expect("recursive paths stay below the package root")
                .components()
                .map(|component| component.as_os_str().to_str())
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Core ML package paths must be valid UTF-8",
                    )
                })?
                .join("/");
            files.push((relative, path));
        }
    }
    Ok(())
}

pub struct CoreMlBackend {
    inner: platform::ModelHandle,
    resolved: ResolvedBackend,
    mapping: CoreMlIoMapping,
    package_identity: CoreMlPackageIdentity,
}

impl CoreMlBackend {
    pub fn load_package(
        package_path: impl AsRef<Path>,
        manifest: &ModelManifest,
        request: &BackendInitRequest,
        timeout: Duration,
    ) -> Result<Self, InitFailure> {
        if timeout.is_zero() {
            return Err(InitFailure::new(
                "native_initialization_timeout_invalid",
                InitializationStage::ModelCompile,
                "Core ML initialization timeout must be greater than zero",
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                BackendKind::CoreMl,
            ));
        }
        let mapping = CoreMlIoMapping::from_manifest(manifest, request)?;
        let package_identity = coreml_package_tree_sha256(&package_path).map_err(|error| {
            InitFailure::new(
                "artifact_integrity_or_target_mismatch",
                InitializationStage::ArtifactIntegrity,
                format!("failed to identify Core ML package tree: {error}"),
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                BackendKind::CoreMl,
            )
        })?;
        if package_identity.tree_sha256 != request.artifact_sha256 {
            return Err(InitFailure::new(
                "artifact_integrity_or_target_mismatch",
                InitializationStage::ArtifactIntegrity,
                format!(
                    "Core ML package tree expected {}, got {}",
                    request.artifact_sha256, package_identity.tree_sha256
                ),
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                BackendKind::CoreMl,
            ));
        }

        let started = Instant::now();
        let inner = platform::load(package_path.as_ref(), request, timeout)?;
        let resolved = ResolvedBackend {
            backend_kind: BackendKind::CoreMl,
            platform: request.target.clone(),
            configured_provider: Some("CoreML".to_owned()),
            accelerator: None,
            execution_plan: ExecutionPlan::Unknown,
            model_version: request.model_version.clone(),
            artifact_id: request.artifact_id.clone(),
            artifact_sha256: request.artifact_sha256.clone(),
            initialization_ms: started.elapsed().as_millis().try_into().unwrap_or(u64::MAX),
            runtime_version: None,
        };
        Ok(Self {
            inner,
            resolved,
            mapping,
            package_identity,
        })
    }

    pub fn resolved_backend(&self) -> &ResolvedBackend {
        &self.resolved
    }

    pub fn io_mapping(&self) -> &CoreMlIoMapping {
        &self.mapping
    }

    pub fn package_identity(&self) -> &CoreMlPackageIdentity {
        &self.package_identity
    }
}

impl RuntimeBackend for CoreMlBackend {
    fn infer(&mut self, input: ModelInput) -> Result<RawModelOutput, InferenceError> {
        input.validate()?;
        platform::infer(&mut self.inner, &self.mapping, input)
    }
}

#[cfg_attr(
    not(all(
        feature = "native-coreml-adapter",
        any(target_os = "macos", target_os = "ios")
    )),
    allow(dead_code)
)]
fn wait_for_initialization<T>(
    receiver: &Receiver<Result<T, String>>,
    timeout: Duration,
    request: &BackendInitRequest,
) -> Result<T, InitFailure> {
    match receiver.recv_timeout(timeout) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(message)) => Err(InitFailure::new(
            "coreml_model_compile_failed",
            InitializationStage::ModelCompile,
            message,
        )
        .with_context(
            request.target.clone(),
            request.model_version.clone(),
            BackendKind::CoreMl,
        )),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(InitFailure::new(
            "native_initialization_timeout",
            InitializationStage::ModelCompile,
            format!("Core ML package compilation exceeded {timeout:?}"),
        )
        .with_context(
            request.target.clone(),
            request.model_version.clone(),
            BackendKind::CoreMl,
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(InitFailure::new(
            "coreml_model_compile_failed",
            InitializationStage::ModelCompile,
            "Core ML package compilation callback disconnected",
        )
        .with_context(
            request.target.clone(),
            request.model_version.clone(),
            BackendKind::CoreMl,
        )),
    }
}

fn manifest_failure(message: impl Into<String>, request: &BackendInitRequest) -> InitFailure {
    InitFailure::new(
        "artifact_integrity_or_target_mismatch",
        InitializationStage::ArtifactIntegrity,
        message,
    )
    .with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::CoreMl,
    )
}

fn io_failure(message: impl Into<String>, request: &BackendInitRequest) -> InitFailure {
    InitFailure::new(
        "coreml_manifest_io_invalid",
        InitializationStage::IoDiscovery,
        message,
    )
    .with_context(
        request.target.clone(),
        request.model_version.clone(),
        BackendKind::CoreMl,
    )
}

#[cfg(all(
    feature = "native-coreml-adapter",
    any(target_os = "macos", target_os = "ios")
))]
mod platform {
    use std::ptr;
    use std::sync::mpsc;

    use block2::RcBlock;
    use objc2::rc::{autoreleasepool, Retained};
    use objc2::runtime::{AnyObject, ProtocolObject};
    use objc2::AnyThread;
    use objc2_core_foundation::CFData;
    use objc2_core_graphics::{
        CGBitmapInfo, CGColorRenderingIntent, CGColorSpace, CGDataProvider, CGImage,
        CGImageAlphaInfo,
    };
    use objc2_core_ml::{
        MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel, MLMultiArray,
        MLMultiArrayDataType,
    };
    use objc2_core_video::kCVPixelFormatType_32RGBA;
    use objc2_foundation::{NSDictionary, NSError, NSString, NSURL};

    use super::*;
    use crate::backend::{RawTensor, TensorData};

    pub struct ModelHandle(Retained<MLModel>);

    // Core ML documents MLModel prediction as safe for concurrent use. The Base
    // RuntimeBackend API still serializes calls through `&mut self`.
    unsafe impl Send for ModelHandle {}

    pub fn load(
        package_path: &Path,
        request: &BackendInitRequest,
        timeout: Duration,
    ) -> Result<ModelHandle, InitFailure> {
        let current = super::super::Platform::current();
        if request.target != current {
            return Err(InitFailure::new(
                "artifact_integrity_or_target_mismatch",
                InitializationStage::RuntimeLoad,
                format!(
                    "Core ML request target {}/{} does not match runner {}/{}",
                    request.target.os, request.target.arch, current.os, current.arch
                ),
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                BackendKind::CoreMl,
            ));
        }
        let path = package_path.to_str().ok_or_else(|| {
            InitFailure::new(
                "coreml_package_path_invalid",
                InitializationStage::ArtifactIntegrity,
                "Core ML package path must be valid UTF-8",
            )
            .with_context(
                request.target.clone(),
                request.model_version.clone(),
                BackendKind::CoreMl,
            )
        })?;

        autoreleasepool(|_| {
            let package_url = NSURL::fileURLWithPath_isDirectory(&NSString::from_str(path), true);
            let (sender, receiver) = mpsc::channel();
            let completion = RcBlock::new(move |compiled: *mut NSURL, error: *mut NSError| {
                let result = unsafe {
                    if let Some(url) = compiled.as_ref() {
                        url.path()
                            .map(|path| path.to_string())
                            .ok_or_else(|| "compiled Core ML URL has no filesystem path".to_owned())
                    } else if let Some(error) = error.as_ref() {
                        Err(error.localizedDescription().to_string())
                    } else {
                        Err("Core ML compilation returned neither a URL nor an error".to_owned())
                    }
                };
                let _ = sender.send(result);
            });
            unsafe {
                MLModel::compileModelAtURL_completionHandler(&package_url, &completion);
            }
            let compiled_path = wait_for_initialization(&receiver, timeout, request)?;
            let compiled_url =
                NSURL::fileURLWithPath_isDirectory(&NSString::from_str(&compiled_path), true);
            let model = unsafe { MLModel::modelWithContentsOfURL_error(&compiled_url) }.map_err(
                |error| {
                    InitFailure::new(
                        "coreml_model_load_failed",
                        InitializationStage::RuntimeLoad,
                        error.localizedDescription().to_string(),
                    )
                    .with_context(
                        request.target.clone(),
                        request.model_version.clone(),
                        BackendKind::CoreMl,
                    )
                },
            )?;
            Ok(ModelHandle(model))
        })
    }

    pub fn infer(
        handle: &mut ModelHandle,
        mapping: &CoreMlIoMapping,
        input: ModelInput,
    ) -> Result<RawModelOutput, InferenceError> {
        let ModelInput::Rgba8 {
            width,
            height,
            bytes,
        } = input
        else {
            return Err(InferenceError::new(
                "unsupported_input",
                "the Core ML image-feature adapter requires RGBA8 input",
            ));
        };
        if width != mapping.input_width || height != mapping.input_height {
            return Err(InferenceError::new(
                "coreml_input_shape_invalid",
                format!(
                    "expected {}x{} RGBA input, got {width}x{height}",
                    mapping.input_width, mapping.input_height
                ),
            ));
        }

        autoreleasepool(|_| unsafe {
            let data = CFData::from_bytes(&bytes);
            let provider = CGDataProvider::with_cf_data(Some(&data)).ok_or_else(|| {
                InferenceError::new("coreml_input_failed", "failed to create CGDataProvider")
            })?;
            let color_space = CGColorSpace::new_device_rgb().ok_or_else(|| {
                InferenceError::new("coreml_input_failed", "failed to create RGB color space")
            })?;
            let image = CGImage::new(
                width as usize,
                height as usize,
                8,
                32,
                width as usize * 4,
                Some(&color_space),
                CGBitmapInfo(CGImageAlphaInfo::Last.0),
                Some(&provider),
                ptr::null(),
                false,
                CGColorRenderingIntent::RenderingIntentDefault,
            )
            .ok_or_else(|| {
                InferenceError::new("coreml_input_failed", "failed to create RGBA CGImage")
            })?;
            let value = MLFeatureValue::featureValueWithCGImage_pixelsWide_pixelsHigh_pixelFormatType_options_error(
                &image,
                width as isize,
                height as isize,
                kCVPixelFormatType_32RGBA,
                None,
            )
            .map_err(|error| {
                InferenceError::new(
                    "coreml_input_failed",
                    error.localizedDescription().to_string(),
                )
            })?;
            let input_name = NSString::from_str(&mapping.input_feature_name);
            let dictionary = NSDictionary::from_slices(&[&*input_name], &[&*value]);
            let dictionary: &NSDictionary<NSString, AnyObject> = dictionary.cast_unchecked();
            let feature_provider = MLDictionaryFeatureProvider::initWithDictionary_error(
                MLDictionaryFeatureProvider::alloc(),
                dictionary,
            )
            .map_err(|error| {
                InferenceError::new(
                    "coreml_input_failed",
                    error.localizedDescription().to_string(),
                )
            })?;
            let input: &ProtocolObject<dyn MLFeatureProvider> =
                ProtocolObject::from_ref(&*feature_provider);
            let output = handle
                .0
                .predictionFromFeatures_error(input)
                .map_err(|error| {
                    InferenceError::new(
                        "inference_failed",
                        error.localizedDescription().to_string(),
                    )
                })?;
            let output_name = NSString::from_str(&mapping.output_feature_name);
            let output_value = output.featureValueForName(&output_name).ok_or_else(|| {
                InferenceError::new(
                    "coreml_output_missing",
                    format!("Core ML did not return {}", mapping.output_feature_name),
                )
            })?;
            let array = output_value.multiArrayValue().ok_or_else(|| {
                InferenceError::new(
                    "coreml_output_type_invalid",
                    "Core ML output is not an MLMultiArray",
                )
            })?;
            let values = copy_contiguous_f32(&array, &mapping.output_shape)?;
            Ok(RawModelOutput {
                tensors: vec![RawTensor {
                    role: mapping.output_role.clone(),
                    shape: mapping.output_shape.clone(),
                    data: TensorData::F32(values),
                }],
            })
        })
    }

    #[allow(deprecated)]
    unsafe fn copy_contiguous_f32(
        array: &MLMultiArray,
        expected_shape: &[usize],
    ) -> Result<Vec<f32>, InferenceError> {
        if array.dataType() != MLMultiArrayDataType::Float32 {
            return Err(InferenceError::new(
                "coreml_output_type_invalid",
                "Core ML output must be FLOAT32",
            ));
        }
        let shape = array.shape();
        let actual_shape = (0..shape.count())
            .map(|index| shape.objectAtIndex(index).unsignedIntegerValue())
            .collect::<Vec<_>>();
        if actual_shape != expected_shape {
            return Err(InferenceError::new(
                "coreml_output_shape_invalid",
                format!("expected {expected_shape:?}, got {actual_shape:?}"),
            ));
        }
        let strides = array.strides();
        let actual_strides = (0..strides.count())
            .map(|index| strides.objectAtIndex(index).unsignedIntegerValue())
            .collect::<Vec<_>>();
        let mut expected_strides = vec![1usize; expected_shape.len()];
        for index in (0..expected_shape.len().saturating_sub(1)).rev() {
            expected_strides[index] = expected_strides[index + 1]
                .checked_mul(expected_shape[index + 1])
                .ok_or_else(|| {
                    InferenceError::new("coreml_output_shape_invalid", "output shape overflows")
                })?;
        }
        if actual_strides != expected_strides {
            return Err(InferenceError::new(
                "coreml_output_layout_unsupported",
                format!("expected contiguous strides {expected_strides:?}, got {actual_strides:?}"),
            ));
        }
        let count = expected_shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or_else(|| {
                InferenceError::new("coreml_output_shape_invalid", "output shape overflows")
            })?;
        if usize::try_from(array.count()).ok() != Some(count) {
            return Err(InferenceError::new(
                "coreml_output_shape_invalid",
                format!("expected {count} output elements, got {}", array.count()),
            ));
        }
        let pointer = array.dataPointer().cast::<f32>().as_ptr();
        let values = std::slice::from_raw_parts(pointer, count).to_vec();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(InferenceError::new(
                "smoke_output_invalid",
                "Core ML returned non-finite FLOAT32 output",
            ));
        }
        Ok(values)
    }
}

#[cfg(not(all(
    feature = "native-coreml-adapter",
    any(target_os = "macos", target_os = "ios")
)))]
mod platform {
    use super::*;

    pub struct ModelHandle;

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    pub fn load(
        _package_path: &Path,
        request: &BackendInitRequest,
        _timeout: Duration,
    ) -> Result<ModelHandle, InitFailure> {
        Err(InitFailure::new(
            "coreml_target_unavailable",
            InitializationStage::RuntimeLoad,
            "Apple Core ML is only loadable on a real macOS or iOS target",
        )
        .with_context(
            request.target.clone(),
            request.model_version.clone(),
            BackendKind::CoreMl,
        ))
    }

    #[cfg(all(
        not(feature = "native-coreml-adapter"),
        any(target_os = "macos", target_os = "ios")
    ))]
    pub fn load(
        _package_path: &Path,
        request: &BackendInitRequest,
        _timeout: Duration,
    ) -> Result<ModelHandle, InitFailure> {
        Err(InitFailure::new(
            "coreml_adapter_feature_disabled",
            InitializationStage::RuntimeLoad,
            "enable native-coreml-adapter to call the Apple Core ML framework",
        )
        .with_context(
            request.target.clone(),
            request.model_version.clone(),
            BackendKind::CoreMl,
        ))
    }

    pub fn infer(
        _handle: &mut ModelHandle,
        _mapping: &CoreMlIoMapping,
        _input: ModelInput,
    ) -> Result<RawModelOutput, InferenceError> {
        Err(InferenceError::new(
            "coreml_target_unavailable",
            "Apple Core ML inference is unavailable in this build",
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    #[test]
    fn initialization_wait_maps_timeout_to_model_compile_stage() {
        let (sender, receiver) = mpsc::channel::<Result<(), String>>();
        let error = wait_for_initialization(&receiver, Duration::from_millis(1), &request())
            .expect_err("sender is alive but does not complete before the deadline");
        drop(sender);
        assert_eq!(error.code.as_ref(), "native_initialization_timeout");
        assert_eq!(error.stage, InitializationStage::ModelCompile);
        assert_eq!(error.attempted_backend, Some(BackendKind::CoreMl));
    }

    fn request() -> BackendInitRequest {
        BackendInitRequest {
            target: super::super::Platform::new("macos", "aarch64"),
            model_id: "rimeflow-yolov8n".to_owned(),
            model_version: "yolov8n-coreml-20260811".to_owned(),
            artifact_id: "apple-coreml-fp32".to_owned(),
            artifact_sha256: "b5c7cf2cf8eb1b6dc313874a84b0006e4b3ef778c0d18bbcb0f06b425fbfd562"
                .to_owned(),
        }
    }
}
