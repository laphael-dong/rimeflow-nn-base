// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use build_print::info;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{self, copy};
use std::path::{Path, PathBuf};
use std::process::Command;

// Constansts that are used by the build script.

// The environment variable that contains the output directory where it's possible to write or modify files.
const OUT_DIR_ENV_VAR: &str = "OUT_DIR";
// Cargo environment variable to definbe target os and architecture.
const CARGO_CFG_TARGET_OS_ENV_VAR: &str = "CARGO_CFG_TARGET_OS";
const CARGO_CFG_TARGET_ARCH_ENV_VAR: &str = "CARGO_CFG_TARGET_ARCH";
// If we are generating docuemntation in hermetic environment for docs.rs
const CARGO_DOCS_RS: &str = "DOCS_RS";

const LITERT_CC_SDK_ENV: &str = "RIMEFLOW_LITERT_CC_SDK_ZIP";
const LITERT_CC_SDK_SHA256_ENV: &str = "RIMEFLOW_LITERT_CC_SDK_SHA256";
const LITERT_RUNTIME_ENV: &str = "RIMEFLOW_LITERT_RUNTIME_SO";
const LITERT_RUNTIME_SHA256_ENV: &str = "RIMEFLOW_LITERT_RUNTIME_SHA256";
const ANDROID_NDK_HOME_ENV: &str = "ANDROID_NDK_HOME";

#[derive(Debug, PartialEq, Clone, Copy)]
enum Platform {
    AndroidArm64,
    AndroidX86,
    LinuxArm64,
    LinuxX86,
    MacosArm64,
    WindowsX86,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum LiteRtAccelerator {
    Metal,
    OpenGL,
    WebGPU,
    None,
}

impl LiteRtAccelerator {
    fn as_str(&self) -> &str {
        match self {
            Self::Metal => "metal",
            Self::OpenGL => "opengl",
            Self::WebGPU => "webgpu",
            Self::None => "none",
        }
    }
}

#[derive(Debug, PartialEq)]
struct TargetPlatform {
    platform: Platform,
    accelerator: LiteRtAccelerator,
}

impl TargetPlatform {
    // Create target platform from cargo environment variables.
    fn from_cargo_env() -> Result<Self, String> {
        let os = env::var(CARGO_CFG_TARGET_OS_ENV_VAR).unwrap_or_else(|_| "unknown".to_string());
        let arch =
            env::var(CARGO_CFG_TARGET_ARCH_ENV_VAR).unwrap_or_else(|_| "unknown".to_string());
        info!("Target os {} platform {}", os.as_str(), arch.as_str());

        // Detect accelerator based on the features in Cargo.toml.
        let accelerator = if env::var("CARGO_FEATURE_METAL").is_ok() {
            LiteRtAccelerator::Metal
        } else if env::var("CARGO_FEATURE_WEBGPU").is_ok() {
            LiteRtAccelerator::WebGPU
        } else if env::var("CARGO_FEATURE_OPENGL").is_ok() {
            LiteRtAccelerator::OpenGL
        } else {
            LiteRtAccelerator::None
        };
        info!("Target accelerator {}", accelerator.as_str());

        match (os.as_str(), arch.as_str(), accelerator.as_str()) {
            ("android", "aarch64", "none") => Ok(TargetPlatform {
                platform: Platform::AndroidArm64,
                accelerator: LiteRtAccelerator::None,
            }),
            ("android", "aarch64", "opengl") => Ok(TargetPlatform {
                platform: Platform::AndroidArm64,
                accelerator: LiteRtAccelerator::OpenGL,
            }),
            ("android", "x86_64", "none") => Ok(TargetPlatform {
                platform: Platform::AndroidX86,
                accelerator: LiteRtAccelerator::None,
            }),
            ("android", "x86_64", "opengl") => Ok(TargetPlatform {
                platform: Platform::AndroidX86,
                accelerator: LiteRtAccelerator::OpenGL,
            }),
            ("linux", "aarch64", "none") => Ok(TargetPlatform {
                platform: Platform::LinuxArm64,
                accelerator: LiteRtAccelerator::None,
            }),
            ("linux", "aarch64", "webgpu") => Ok(TargetPlatform {
                platform: Platform::LinuxArm64,
                accelerator: LiteRtAccelerator::WebGPU,
            }),
            ("linux", "x86_64", "none") => Ok(TargetPlatform {
                platform: Platform::LinuxX86,
                accelerator: LiteRtAccelerator::None,
            }),
            ("linux", "x86_64", "webgpu") => Ok(TargetPlatform {
                platform: Platform::LinuxX86,
                accelerator: LiteRtAccelerator::WebGPU,
            }),
            ("macos", "aarch64", "none") => Ok(TargetPlatform {
                platform: Platform::MacosArm64,
                accelerator: LiteRtAccelerator::None,
            }),
            ("macos", "aarch64", "metal") => Ok(TargetPlatform {
                platform: Platform::MacosArm64,
                accelerator: LiteRtAccelerator::Metal,
            }),
            ("windows", "x86_64", "none") => Ok(TargetPlatform {
                platform: Platform::WindowsX86,
                accelerator: LiteRtAccelerator::None,
            }),
            ("windows", "x86_64", "webgpu") => Ok(TargetPlatform {
                platform: Platform::WindowsX86,
                accelerator: LiteRtAccelerator::WebGPU,
            }),
            _ => Err(format!(
                "Unknown target platform os:{} aarch:{} accelerator:{}",
                os,
                arch,
                accelerator.as_str()
            )),
        }
    }

    // See https://ai.google.dev/edge/litert/next/cpp_sdk
    fn runtime_name(&self) -> String {
        match self.platform {
            Platform::AndroidArm64
            | Platform::AndroidX86
            | Platform::LinuxArm64
            | Platform::LinuxX86 => "libLiteRt.so".to_string(),
            Platform::MacosArm64 => "libLiteRt.dylib".to_string(),
            Platform::WindowsX86 => "libLiteRt.dll".to_string(),
        }
    }

    fn accelerator_runtime_name(&self) -> Option<String> {
        match (self.platform, self.accelerator) {
            (Platform::AndroidArm64, LiteRtAccelerator::OpenGL) => {
                Some("libLiteRtClGlAccelerator.so".to_string())
            }
            (Platform::AndroidX86, LiteRtAccelerator::OpenGL) => {
                Some("libLiteRtClGlAccelerator.so".to_string())
            }
            (Platform::LinuxArm64, LiteRtAccelerator::WebGPU) => {
                Some("libLiteRtWebGpuAccelerator.so".to_string())
            }
            (Platform::LinuxX86, LiteRtAccelerator::WebGPU) => {
                Some("libLiteRtWebGpuAccelerator.so".to_string())
            }
            (Platform::MacosArm64, LiteRtAccelerator::Metal) => {
                Some("libLiteRtMetalAccelerator.dylib".to_string())
            }
            (Platform::WindowsX86, LiteRtAccelerator::WebGPU) => {
                Some("libLiteRtWebGpuAccelerator.dll".to_string())
            }
            _ => None,
        }
    }

    fn runtime_link_name(&self) -> String {
        "LiteRt".to_string()
    }

}

// Helper function to check if a tool is installed
fn check_tool_installed(tool: &str) -> Result<(), String> {
    match Command::new(tool).arg("--version").output() {
        Ok(output) if output.status.success() => Ok(()),
        _ => Err(format!("Required tool '{}' is not installed or not in PATH.", tool)),
    }
}

fn verified_copy(
    source_env: &str,
    sha256_env: &str,
    destination: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    let source = PathBuf::from(env::var(source_env)?);
    let expected = env::var(sha256_env)?;
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{} is not a lowercase SHA-256", sha256_env).into());
    }
    let bytes = fs::read(&source)?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != expected {
        return Err(format!(
            "verified LiteRT input {} digest mismatch: expected {}, got {}",
            source.display(),
            expected,
            actual
        )
        .into());
    }
    fs::write(destination, bytes)?;
    Ok(())
}

fn unzip_archive(
    archive_path: &Path,
    extract_to: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let file = File::open(archive_path)?;
    let mut archive = zip::ZipArchive::new(file)?;
    // Assume that the directory of the archive has one top directory, that the function return.
    let mut root_directory = PathBuf::new();

    for i in 0..archive.len() {
        let mut file = archive.by_index(i)?;
        let file_name = file.mangled_name();
        let outpath = extract_to.join(&file_name);

        // Extract the root directory of the file.
        if let Some(rd) = Path::new(&file_name).iter().next() {
            root_directory = Path::new(rd).to_path_buf();
        }
        if file.name().ends_with('/') {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
        }
    }
    Ok(root_directory)
}

fn stage_runtime(
    target_plarform: &TargetPlatform,
    out_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Staging verified LiteRT runtime");

    let runtime_name = target_plarform.runtime_name();
    let runtime_local_name = out_dir.join(runtime_name);
    verified_copy(
        LITERT_RUNTIME_ENV,
        LITERT_RUNTIME_SHA256_ENV,
        &runtime_local_name,
    )?;
    return Ok(());
}

struct LiteRTSdk {
    sdk_root_dir: PathBuf,
    sdk_build_dir: PathBuf,
}

impl LiteRTSdk {
    fn new(sdk_root_dir: PathBuf, sdk_build_dir: PathBuf) -> Self {
        Self { sdk_root_dir, sdk_build_dir }
    }
}

// Following LiteRT C++ SDK build process.
// https://ai.google.dev/edge/litert/next/cpp_sdk
fn stage_and_build_cpp_sdk(
    target_platform: &TargetPlatform,
    out_dir: &Path,
) -> Result<LiteRTSdk, Box<dyn std::error::Error>> {
    info!("Staging verified LiteRT C++ SDK...");
    let sdk_zip_path = out_dir.join("litert_cc_sdk.zip");
    verified_copy(LITERT_CC_SDK_ENV, LITERT_CC_SDK_SHA256_ENV, &sdk_zip_path)?;

    info!("Unzipping LiteRT C++ SDK...");
    let sdk_extract_dir = out_dir.join("litert_cc_sdk_extracted");
    if !sdk_extract_dir.exists() {
        fs::create_dir_all(&sdk_extract_dir)?;
    }
    let zip_root_dir = unzip_archive(&sdk_zip_path, &sdk_extract_dir)?;
    let sdk_root = sdk_extract_dir.join(zip_root_dir);

    info!("Staging verified runtime for SDK to {}", sdk_root.display());
    stage_runtime(target_platform, &sdk_root)?;

    let build_dir = out_dir.join("litert_cc_sdk_build");
    info!("Building C++ SDK with CMake in {}", build_dir.display());

    let ndk = PathBuf::from(env::var(ANDROID_NDK_HOME_ENV)?);
    let llvm = ndk.join("toolchains/llvm/prebuilt/linux-x86_64");
    let c_compiler = llvm.join("bin/aarch64-linux-android26-clang");
    let cxx_compiler = llvm.join("bin/aarch64-linux-android26-clang++");
    let toolchain = ndk.join("build/cmake/android.toolchain.cmake");
    for required in [&c_compiler, &cxx_compiler, &toolchain] {
        if !required.is_file() {
            return Err(format!("NDK r27c tool is missing: {}", required.display()).into());
        }
    }

    let status = Command::new("cmake")
        .arg("-S")
        .arg(&sdk_root)
        .arg("-B")
        .arg(&build_dir)
        .arg(format!("-DCMAKE_TOOLCHAIN_FILE={}", toolchain.display()))
        .arg("-DANDROID_ABI=arm64-v8a")
        .arg("-DANDROID_PLATFORM=android-26")
        .arg("-DANDROID_STL=c++_static")
        .arg(format!("-DCMAKE_C_COMPILER={}", c_compiler.display()))
        .arg(format!("-DCMAKE_CXX_COMPILER={}", cxx_compiler.display()))
        .status()?;

    if !status.success() {
        return Err(format!("CMake configure failed with status: {}", status).into());
    }

    let status = Command::new("cmake")
        .arg("--build")
        .arg(&build_dir)
        .arg("--target")
        .arg("litert_cc_api")
        .arg("-j")
        .status()?;

    if !status.success() {
        return Err(format!("CMake build failed with status: {}", status).into());
    }

    Ok(LiteRTSdk::new(sdk_root, build_dir))
}

// The wrapper.h is moved to output directory, to avoid include of local files when
// cargo is called from the source tree.
fn copy_wrapper_h(out_dir: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let wrapper_h = "wrapper.h";
    let wrapper_h_path = out_dir.join(wrapper_h);
    let mut outfile = File::create(&wrapper_h_path)?;
    let mut infile = File::open(wrapper_h)?;
    copy(&mut infile, &mut outfile)?;
    Ok(wrapper_h_path)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo::rustc-check-cfg=cfg(bazel_bindgen, cargo_bindgen, docsrs, async_support)");
    // Check if we are currently generating documentation
    let is_doc_gen = env::var(CARGO_DOCS_RS).is_ok();

    if is_doc_gen {
        info!("Skipping heavy lifting because we are just building docs!");
        println!("cargo::rustc-cfg=docsrs");
        // Force async support for docs.rs
        println!("cargo::rustc-cfg=async_support");
        return Ok(());
    }
    println!("cargo::rerun-if-changed=build/build.rs");
    println!("cargo::rerun-if-changed=wrapper.h");
    for variable in [
        LITERT_CC_SDK_ENV,
        LITERT_CC_SDK_SHA256_ENV,
        LITERT_RUNTIME_ENV,
        LITERT_RUNTIME_SHA256_ENV,
        ANDROID_NDK_HOME_ENV,
    ] {
        println!("cargo::rerun-if-env-changed={variable}");
    }
    if env::var("CARGO_FEATURE_ASYNC_SUPPORT").is_ok() {
        println!("cargo::rustc-cfg=async_support");
    }

    if let Ok(manifest_dir) = env::var("CARGO_MANIFEST_DIR") {
        info!("Manifest dir {}", manifest_dir);
    }

    let target_platform = TargetPlatform::from_cargo_env()?;

    check_tool_installed("cmake")?;

    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let litert_sdk = stage_and_build_cpp_sdk(&target_platform, &out_dir)?;
    info!("Runtime dir {}", litert_sdk.sdk_root_dir.display());
    info!("Build dir {}", litert_sdk.sdk_build_dir.display());

    println!("cargo::rustc-link-search=native={}", litert_sdk.sdk_root_dir.display());
    println!("cargo::rustc-link-lib=dylib={}", target_platform.runtime_link_name());
    println!("cargo::rustc-link-search=native={}", litert_sdk.sdk_build_dir.display());
    println!("cargo::rustc-link-lib=static=litert_cc_api");

    if let Some(accel_runtime_name) = target_platform.accelerator_runtime_name() {
        println!("cargo::rustc-link-lib=dylib={}", accel_runtime_name);
    }

    let wrapper_h_path = copy_wrapper_h(&out_dir)?;
    let bindings = bindgen::Builder::default()
        .header(wrapper_h_path.to_str().unwrap())
        // Add the include path so clang can find dependent headers
        .clang_arg(format!("-I{}", litert_sdk.sdk_root_dir.display()))
        .clang_arg(format!("-I{}", litert_sdk.sdk_root_dir.join("litert").join("c").display()))
        .clang_arg(format!("-I{}", litert_sdk.sdk_build_dir.join("include").display()))
        .clang_arg("--target=aarch64-linux-android26")
        .clang_arg(format!(
            "--sysroot={}",
            PathBuf::from(env::var(ANDROID_NDK_HOME_ENV)?)
                .join("toolchains/llvm/prebuilt/linux-x86_64/sysroot")
                .display()
        ))
        .clang_arg("-DLITERT_DISABLE_GPU")
        .layout_tests(false)
        .derive_default(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    // Get the output directory where cargo wants us to build things
    let out_dir = PathBuf::from(env::var(OUT_DIR_ENV_VAR)?);
    let bindings_out_path = out_dir.join("bindings.rs");
    info!("Writing binding.rs to {}", bindings_out_path.display());
    bindings.write_to_file(bindings_out_path).expect("Couldn't write bindings!");
    println!("cargo::rustc-cfg=cargo_bindgen");

    Ok(())
}
