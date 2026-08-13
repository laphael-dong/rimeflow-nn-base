import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const checkedOn = '2026-08-10';
const matrixPath = resolve(root, 'evidence/platform/platform-matrix.json');
const matrix = JSON.parse(await readFile(matrixPath, 'utf8'));
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));
const conversion = publication.operatorInputPublication.objects.find((object) => object.id === 'conversion-summary');
const conversionReportSha256 = conversion.sha256;
const sources = {
  apple: {
    minimumOs: ['https://apple.github.io/coremltools/docs-guides/source/deployment-compatibility.html', '最低 OS 是项目验收下限；Core ML artifact 的实际 deployment target 还必须由 Apple runner 产物证明。'],
    sdkRuntime: ['https://developer.apple.com/support/xcode/', 'Xcode/系统 Core ML 版本必须由目标 runner 的 xcodebuild 与系统版本采集。'],
    converter: ['https://pypi.org/project/coremltools/9.0/', '任务 1 Python 工具环境固定 coremltools 9.0；它不提供当前 ONNX 的直接转换链。'],
  },
  android: {
    minimumOs: ['https://developer.android.com/tools/releases/platforms', 'API 26 是项目接受下限；真实 CompiledModel accelerator 能力仍按设备探测。'],
    sdkRuntime: ['https://ai.google.dev/edge/litert/next/get_started', 'LiteRT v2 runtime 固定为 PyPI/Android artifact 的 2.1.6 线，设备侧版本必须同报告。'],
    converter: ['https://ai.google.dev/edge/litert/models/convert', 'LiteRT 接受 TFLite；ONNX 不是 ai-edge-litert runtime 包提供的转换入口。'],
  },
  windows: {
    minimumOs: ['https://learn.microsoft.com/windows/apps/windows-app-sdk/system-requirements', 'Windows 11 是项目验收下限，x64 与 ARM64 必须各自真实加载。'],
    sdkRuntime: ['https://learn.microsoft.com/windows/ai/new-windows-ml/overview', 'Windows ML 随 Windows App SDK；版本必须由 Windows runner 的项目锁和加载日志证明。'],
    converter: ['https://learn.microsoft.com/windows/ai/new-windows-ml/get-started', 'Windows ML 直接加载 ONNX，无格式转换；必须实际 Load/Run，不能生成虚构 artifact。'],
  },
  linux: {
    minimumOs: ['https://onnxruntime.ai/docs/install/', 'glibc 2.35/Ubuntu 22.04 是项目接受下限，不宣称为 ORT 的全局最低要求。'],
    sdkRuntime: ['https://onnxruntime.ai/docs/execution-providers/', '每个 EP 单独请求并记录实际 inference，CPU 成功不能推断 OpenVINO/CUDA/TensorRT。'],
    converter: ['https://onnxruntime.ai/docs/', '原 ONNX 无格式转换。'],
  },
  harmonyos: {
    minimumOs: ['https://developer.huawei.com/consumer/en/doc/harmonyos-releases-V5/', 'HarmonyOS NEXT 5.0 是项目验收线，需真实 arm64 设备。'],
    sdkRuntime: ['https://www.mindspore.cn/lite/docs/en/r2.7.0/index.html', 'MindSpore Lite 2.7.0 与 converter_lite 必须同版本。'],
    converter: ['https://www.mindspore.cn/lite/docs/en/r2.7.0/use/converter_tool.html', '使用官方 converter_lite ONNX 入口并记录实际算子/I/O。'],
  },
  web: {
    minimumOs: ['https://onnxruntime.ai/docs/tutorials/web/env-flags-and-session-options.html', '浏览器版本是项目测试下限，不等同于 ORT 官方长期支持承诺。'],
    sdkRuntime: ['https://www.npmjs.com/package/onnxruntime-web/v/1.27.0', 'WASM/WebGPU 均固定 onnxruntime-web 1.27.0，EP 数值与性能分开。'],
    converter: ['https://onnxruntime.ai/docs/tutorials/web/', 'Web 直接加载 ONNX，无格式转换。'],
  },
};
for (const platform of matrix.platforms) {
  const key = platform.os === 'macos' || platform.os === 'ios' ? 'apple' : platform.os;
  const source = sources[key];
  platform.officialVersionEvidence = {
    checkedOn,
    minimumOs: { value: platform.minimumOsVersion, officialSource: source.minimumOs[0], selectionBasis: source.minimumOs[1] },
    sdkRuntime: { value: platform.officialSdkRuntime, officialSource: source.sdkRuntime[0], selectionBasis: source.sdkRuntime[1] },
    converter: { value: platform.converter, officialSource: source.converter[0], selectionBasis: source.converter[1] },
  };
  platform.operatorConversionEvidence = {
    path: 'evidence/conversions/conversion-spikes.json',
    blob: conversion.blob,
    sha256: conversionReportSha256,
  };
  platform.requiredCiState = publication.requiredCiState;
}
const linuxCpu = matrix.platforms.find((item) => item.id === 'linux-x86_64-ort-cpu');
const harmony = matrix.platforms.find((item) => item.id === 'harmonyos-arm64-mindspore');
if (!linuxCpu || !harmony) throw new Error('platform matrix missing Linux/Harmony records');
linuxCpu.evidence = `operator conversion report ${conversionReportSha256} 使用固定 canonical 输入完成 onnxruntime-node CPU inference；base 性能报告另以 exact NativeOrtBackend 与 Web WASM 做同机比较，尚无 adapter/package conformance`;
harmony.evidence = '官方 converter_lite 2.7.0 已实际进入 ONNX parse/graph optimization，并在 /model.22/dfl/conv/Conv 的 Conv2DFusion infer-shape 失败；无转换 artifact、HarmonyOS SDK/runner/真实设备';
await writeFile(matrixPath, `${JSON.stringify(matrix, null, 2)}\n`);
const environment = {
  schemaVersion: 2,
  checkedOn,
  classification: 'historical-measurement-environment',
  host: { os: 'linux', arch: 'x64', kernel: '7.0.0-28-generic', cpu: 'AMD Ryzen 7 8840HS w/ Radeon 780M Graphics', logicalCpuCount: 16, memoryBytes: 32883466240 },
  toolchains: {
    rust: 'rustc 1.97.1 (8bab26f4f 2026-07-14)', cargo: 'cargo 1.97.1 (c980f4866 2026-06-30)', bun: '1.3.14', node: 'v24.15.0',
    androidSdk: { state: 'not-observed', value: null }, androidNdk: { state: 'not-observed', value: null, contractRequirement: 'Android platform contract requires NDK r27c on its future target runner; this Linux historical measurement did not observe it.' },
    adb: { state: 'observed', value: 'List of devices attached' }, androidDevice: 'none', apple: 'unavailable: ENOENT', windows: 'unavailable: ENOENT', harmonyos: 'unavailable: ENOENT', cuda: 'unavailable: ENOENT', openvino: 'unavailable: ENOENT',
  },
  evidenceLimit: '本机只能提供 Linux x86_64/Node-WASM 证据；其他平台不能标记 supported。',
};
await writeFile(resolve(root, 'evidence/reports/local-environment.json'), `${JSON.stringify(environment, null, 2)}\n`);
console.log(JSON.stringify({ matrix: 'evidence/platform/platform-matrix.json', environment: 'evidence/reports/local-environment.json', platformCount: matrix.platforms.length }));
