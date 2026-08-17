#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cache_root="${RIMEFLOW_ANDROID_CACHE:-${repo_root}/.android-litert}"
downloads="${cache_root}/downloads"
ndk_root="${cache_root}/android-ndk-r27c"
mkdir -p "${downloads}"

expected_rustc="rustc 1.97.1 (8bab26f4f 2026-07-14)"
expected_cargo="cargo 1.97.1 (c980f4866 2026-06-30)"
[[ "$(rustc --version)" == "${expected_rustc}" ]] || { printf '需要 %s\n' "${expected_rustc}" >&2; exit 1; }
[[ "$(cargo --version)" == "${expected_cargo}" ]] || { printf '需要 %s\n' "${expected_cargo}" >&2; exit 1; }

ndk_url="https://dl.google.com/android/repository/android-ndk-r27c-linux.zip"
ndk_sha256="59c2f6dc96743b5daf5d1626684640b20a6bd2b1d85b13156b90333741bad5cc"
sdk_url="https://github.com/google-ai-edge/LiteRT/releases/download/v2.1.6/litert_cc_sdk.zip"
sdk_sha256="2cbde8fc18cd3d6ffbab6bcdecb92b1d49b198e50a7bdf46e01cd329c657aca8"
runtime_url="https://storage.googleapis.com/litert/binaries/2.1.6/android_arm64/libLiteRt.so"
runtime_sha256="35e34acfb76722868b0fe6bccab9d4432ac3f9fe95e7f29d2d6c030b66052369"

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
fetch_verified "${runtime_url}" "${runtime_sha256}" "${downloads}/libLiteRt-2.1.6-android-arm64.so"

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
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="${llvm}/bin/aarch64-linux-android26-clang"
export CC_aarch64_linux_android="${llvm}/bin/aarch64-linux-android26-clang"
export CXX_aarch64_linux_android="${llvm}/bin/aarch64-linux-android26-clang++"
export AR_aarch64_linux_android="${llvm}/bin/llvm-ar"
export RIMEFLOW_LITERT_CC_SDK_ZIP="${downloads}/litert_cc_sdk-2.1.6.zip"
export RIMEFLOW_LITERT_CC_SDK_SHA256="${sdk_sha256}"
export RIMEFLOW_LITERT_RUNTIME_SO="${downloads}/libLiteRt-2.1.6-android-arm64.so"
export RIMEFLOW_LITERT_RUNTIME_SHA256="${runtime_sha256}"
export CARGO_TARGET_DIR="${cache_root}/target"
export CARGO_HOME="${cache_root}/cargo-home"

build_provenance="${cache_root}/build-provenance.json"
printf '{\n  "schemaVersion": 1,\n  "target": "android-arm64-v8a-api26-cpu",\n  "ndk": {"revision": "27.2.12479018", "url": "%s", "sha256": "%s"},\n  "litertSdk": {"version": "2.1.6", "url": "%s", "sha256": "%s"},\n  "litertRuntime": {"version": "2.1.6", "url": "%s", "sha256": "%s"},\n  "rustBinding": {"crate": "google-ai-edge-litert", "version": "0.1.3", "upstreamCrateSha256": "fe78e8555c7cc89d78e92b06b976e049f793ef38d6962d5a0354794650bc23f8"},\n  "rustc": "%s",\n  "cargo": "%s",\n  "cmake": "%s"\n}\n' \
  "${ndk_url}" "${ndk_sha256}" "${sdk_url}" "${sdk_sha256}" "${runtime_url}" "${runtime_sha256}" \
  "$(rustc --version)" "$(cargo --version)" "$(cmake --version | head -n1)" > "${build_provenance}"

cd "${repo_root}"
cargo build --locked --release --target aarch64-linux-android --features android-litert-runner --bin rimeflow-android-litert-runner

if [[ "$#" -gt 0 ]]; then
  node tools/android-litert-runner/package_bundle.mjs \
    --runner "${CARGO_TARGET_DIR}/aarch64-linux-android/release/rimeflow-android-litert-runner" \
    --runtime "${RIMEFLOW_LITERT_RUNTIME_SO}" \
    --build-provenance "${build_provenance}" \
    "$@"
fi

printf 'rustc=%s\n' "$(rustc --version)"
printf 'ndk=%s\n' "$(awk -F' = ' '/Pkg.Revision/ { print $2; exit }' "${ndk_root}/source.properties")"
printf 'litert-runtime=%s sha256=%s\n' "2.1.6" "${runtime_sha256}"
printf 'litert-rust-binding=%s\n' "0.1.3"
