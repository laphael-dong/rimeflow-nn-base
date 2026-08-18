#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${RIMEFLOW_ANDROID_CACHE:-${repo_root}/.android-litert}"
downloads="${cache_root}/downloads"
ndk_root="${cache_root}/android-ndk-r27c"

requested_arch="${RIMEFLOW_ANDROID_ARCH:-arm64}"
print_contract=false
package_args=()
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --arch)
      [[ "$#" -ge 2 ]] || { printf '%s\n' '--arch 需要值' >&2; exit 2; }
      requested_arch="$2"
      shift 2
      ;;
    --print-contract)
      print_contract=true
      shift
      ;;
    *)
      package_args+=("$1")
      shift
      ;;
  esac
done

ndk_url="https://dl.google.com/android/repository/android-ndk-r27c-linux.zip"
ndk_sha256="59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc"
sdk_url="https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.6/litert_cc_sdk.zip"
sdk_sha256="2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8"
runtime_url="https://storage.googleapis.com/litert/binaries/2.1.6/android_arm64/libLiteRt.so"
runtime_sha256="35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369"

case "${requested_arch}" in
  arm64)
    cargo_target="aarch64-linux-android"
    compiler_target="aarch64-linux-android26"
    cmake_abi="arm64-v8a"
    bundle_arch="arm64"
    provenance_target="android-arm64-v8a-api26-cpu"
    runtime_file="libLiteRt-2.1.6-android-arm64.so"
    ;;
  x86_64)
    cargo_target="x86_64-linux-android"
    compiler_target="x86_64-linux-android26"
    cmake_abi="x86_64"
    bundle_arch="x86_64"
    provenance_target="android-x86_64-api26-cpu"
    runtime_url="https://storage.googleapis.com/litert/binaries/2.1.6/android_x86_64/libLiteRt.so"
    runtime_sha256="aa1530ba8b37b537d37139760716d183d2d7dc1f7781791ddf1d071c73eca535"
    runtime_file="libLiteRt-2.1.6-android-x86_64.so"
    ;;
  *)
    printf '不支持 Android LiteRT arch：%s（仅允许 arm64 或 x86_64）\n' "${requested_arch}" >&2
    exit 2
    ;;
esac

if [[ "${print_contract}" == true ]]; then
  printf '{"schemaVersion":1,"arch":"%s","cargoTarget":"%s","compilerTarget":"%s","cmakeAbi":"%s","bundleArch":"%s","provenanceTarget":"%s","minimumApi":26,"cpuOnly":true,"litertRuntime":{"version":"2.1.6","url":"%s","sha256":"%s"}}\n' \
    "${requested_arch}" "${cargo_target}" "${compiler_target}" "${cmake_abi}" "${bundle_arch}" "${provenance_target}" "${runtime_url}" "${runtime_sha256}"
  exit 0
fi

expected_rustc="rustc 1.97.1 (8bab26f4f 2026-07-14)"
expected_cargo="cargo 1.97.1 (c980f4866 2026-06-30)"
[[ "$(rustc --version)" == "${expected_rustc}" ]] || { printf '需要 %s\n' "${expected_rustc}" >&2; exit 1; }
[[ "$(cargo --version)" == "${expected_cargo}" ]] || { printf '需要 %s\n' "${expected_cargo}" >&2; exit 1; }

mkdir -p "${downloads}"

fetch_verified() {
  local url="$1" expected="$2" destination="$3"
  if [[ -f "${destination}" ]] && [[ "$(sha256sum "${destination}" | cut -d' ' -f1)" == "${expected}" ]]; then
    return
  fi
  curl --fail --location --retry 3 --output "${destination}.partial" "${url}"
  printf '%s  %s\n' "${expected}" "${destination}.partial" | sha256sum --check --status
  mv "${destination}.partial" "${destination}"
}

fetch_verified "${ndk_url}" "${ndk_sha256}" "${downloads}/android-ndk-r27c-linux.zip"
fetch_verified "${sdk_url}" "${sdk_sha256}" "${downloads}/litert_cc_sdk-2.1.6.zip"
fetch_verified "${runtime_url}" "${runtime_sha256}" "${downloads}/${runtime_file}"

if [[ ! -f "${ndk_root}/source.properties" ]] || ! grep -q 'Pkg.Revision = 27.2.12479018' "${ndk_root}/source.properties"; then
  extract_root="${cache_root}/ndk-extract"
  mkdir -p "${extract_root}"
  unzip -q "${downloads}/android-ndk-r27c-linux.zip" -d "${extract_root}"
  mv "${extract_root}/android-ndk-r27c" "${ndk_root}"
fi

llvm="${ndk_root}/toolchains/llvm/prebuilt/linux-x86_64"
export ANDROID_NDK_HOME="${ndk_root}"
export ANDROID_NDK_ROOT="${ndk_root}"
export LIBCLANG_PATH="${llvm}/musl/lib"
cargo_target_env="${cargo_target^^}"
cargo_target_env="${cargo_target_env//-/_}"
cc_target_env="${cargo_target//-/_}"
export "CARGO_TARGET_${cargo_target_env}_LINKER=${llvm}/bin/${compiler_target}-clang"
export "CC_${cc_target_env}=${llvm}/bin/${compiler_target}-clang"
export "CXX_${cc_target_env}=${llvm}/bin/${compiler_target}-clang++"
export "AR_${cc_target_env}=${llvm}/bin/llvm-ar"
export RIMEFLOW_LITERT_CC_SDK_ZIP="${downloads}/litert_cc_sdk-2.1.6.zip"
export RIMEFLOW_LITERT_CC_SDK_SHA256="${sdk_sha256}"
export RIMEFLOW_LITERT_RUNTIME_SO="${downloads}/${runtime_file}"
export RIMEFLOW_LITERT_RUNTIME_SHA256="${runtime_sha256}"
export CARGO_TARGET_DIR="${cache_root}/target"
export CARGO_HOME="${cache_root}/cargo-home"

build_provenance="${cache_root}/build-provenance-${bundle_arch}.json"
printf '{\n  "schemaVersion": 1,\n  "target": "%s",\n  "arch": "%s",\n  "cargoTarget": "%s",\n  "compilerTarget": "%s",\n  "cmakeAbi": "%s",\n  "ndk": {"revision": "27.2.12479018", "url": "%s", "sha256": "%s"},\n  "litertSdk": {"version": "2.1.6", "url": "%s", "sha256": "%s"},\n  "litertRuntime": {"version": "2.1.6", "url": "%s", "sha256": "%s"},\n  "rustBinding": {"crate": "google-ai-edge-litert", "version": "0.1.3", "upstreamCrateSha256": "fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8"},\n  "rustc": "%s",\n  "cargo": "%s",\n  "cmake": "%s"\n}\n' \
  "${provenance_target}" "${bundle_arch}" "${cargo_target}" "${compiler_target}" "${cmake_abi}" \
  "${ndk_url}" "${ndk_sha256}" "${sdk_url}" "${sdk_sha256}" "${runtime_url}" "${runtime_sha256}" \
  "$(rustc --version)" "$(cargo --version)" "$(cmake --version | head -n1)" > "${build_provenance}"

cd "${repo_root}"
cargo build --locked --release --target "${cargo_target}" --features android-litert-runner --bin rimeflow-android-litert-runner

if [[ "${#package_args[@]}" -gt 0 ]]; then
  node tools/android-litert-runner/package_bundle.mjs \
    --runner "${CARGO_TARGET_DIR}/${cargo_target}/release/rimeflow-android-litert-runner" \
    --runtime "${RIMEFLOW_LITERT_RUNTIME_SO}" \
    --build-provenance "${build_provenance}" \
    "${package_args[@]}"
fi

printf 'rustc=%s\n' "$(rustc --version)"
printf 'ndk=%s\n' "$(awk -F' = ' '/Pkg.Revision/ { print $2; exit }' "${ndk_root}/source.properties")"
printf 'litert-runtime=%s sha256=%s\n' "2.1.6" "${runtime_sha256}"
printf 'litert-rust-binding=%s\n' "0.1.3"
printf 'android-target=%s cargo=%s cmake-abi=%s compiler=%s\n' "${provenance_target}" "${cargo_target}" "${cmake_abi}" "${compiler_target}"
