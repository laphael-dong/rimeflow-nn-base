import { createHash } from 'node:crypto';
import { copyFile, mkdir, readFile, readdir, rename, stat, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { performance } from 'node:perf_hooks';
import { spawnSync } from 'node:child_process';

import {
  OUTPUT_DTYPE,
  OUTPUT_ELEMENTS,
  OUTPUT_NAME,
  OUTPUT_ROLE,
  OUTPUT_SHAPE,
  compareDetections,
  computeNumericalComparison,
  runProductionPostprocess,
  writeF32Artifact,
} from './performance_validation.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const operatorRoot = resolve(process.env.RIMEFLOW_OPERATOR_ROOT ?? resolve(root, '../rimeflow-yolov8n'));
const scratch = resolve(root, '.evidence/task1-linux-baseline');
const outputPath = resolve(root, process.argv[2] ?? '.evidence/task1-linux-baseline/latest-capture.json');
const committedCapturePath = resolve(root, 'evidence/performance/linux-x86_64-capture.json');
if (outputPath === committedCapturePath && process.env.RIMEFLOW_RECORD_BASELINE !== '1') {
  throw new Error('tracked capture record mode requires RIMEFLOW_RECORD_BASELINE=1');
}
const artifactDirectory = outputPath === committedCapturePath
  ? resolve(root, 'evidence/performance/artifacts')
  : resolve(dirname(outputPath), 'artifacts');
const baseEvidenceParentCommit = 'ec3c43a28bc01dba11454d738234ed8fcdd8ba51';
const operatorEvidenceParentCommit = '7f62f8a8e5c69a4f4c880c8bab979f033e7260b9';
await mkdir(scratch, { recursive: true });
await mkdir(dirname(outputPath), { recursive: true });
await mkdir(artifactDirectory, { recursive: true });

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const round = (value, digits = 6) => Number(value.toFixed(digits));
const percentile = (values, fraction) => [...values].sort((a, b) => a - b)[Math.ceil((values.length - 1) * fraction)];
async function directoryBytes(path) { let total = 0; for (const name of await readdir(path)) { const child = resolve(path, name); const info = await stat(child); total += info.isDirectory() ? await directoryBytes(child) : info.size; } return total; }
async function findFiles(path, name, output = []) { for (const item of await readdir(path, { withFileTypes: true })) { const child = resolve(path, item.name); if (item.isDirectory()) await findFiles(child, name, output); else if (item.name === name) output.push(child); } return output; }
async function peakRssBytes() { const status = await readFile(`/proc/${process.pid}/status`, 'utf8'); const line = status.split('\n').find((item) => item.startsWith('VmHWM:')); return Number(line.split(/\s+/)[1]) * 1024; }
function assertFiniteTensor(tensor, expectedShape) {
  if (tensor.type !== OUTPUT_DTYPE || JSON.stringify(tensor.dims) !== JSON.stringify(expectedShape)) throw new Error(`unexpected output contract: ${tensor.type} ${JSON.stringify(tensor.dims)}`);
  if (tensor.data.length !== OUTPUT_ELEMENTS) throw new Error(`unexpected output length: ${tensor.data.length}`);
  for (let index = 0; index < tensor.data.length; index += 1) if (!Number.isFinite(tensor.data[index])) throw new Error(`Web output contains non-finite value at index ${index}`);
}

const preprocessModule = await import(pathToFileURL(resolve(operatorRoot, 'evidence/scripts/preprocess_contract.mjs')));
const ort = await import(pathToFileURL(resolve(operatorRoot, 'evidence/tooling/web/node_modules/onnxruntime-web/dist/ort.node.min.mjs')));
const manifest = JSON.parse(await readFile(resolve(operatorRoot, 'evidence/fixtures/manifest.json'), 'utf8'));
const webReference = JSON.parse(await readFile(resolve(operatorRoot, 'evidence/golden/web-reference.json'), 'utf8'));
const fixture = manifest.images.find((item) => item.id === 'single-target');
if (!fixture) throw new Error('single-target fixture missing');
const image = preprocessModule.readPpm(await readFile(resolve(operatorRoot, fixture.path)));
const canonical = preprocessModule.preprocessCanonical(image);
if (canonical.tensor.length !== 1 * 3 * 640 * 640 || canonical.tensor.some((value) => !Number.isFinite(value))) throw new Error('canonical input contract failed');
const canonicalDigest = preprocessModule.tensorDigest(canonical.tensor);
const inputPath = resolve(scratch, 'single-target.nchw-f32le.bin');
await writeFile(inputPath, Buffer.from(canonical.tensor.buffer, canonical.tensor.byteOffset, canonical.tensor.byteLength));
const modelPath = resolve(operatorRoot, 'models/yolov8n.onnx');
const modelBytes = await readFile(modelPath);

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
const webInitStart = performance.now();
const webSession = await ort.InferenceSession.create(modelBytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });
const webInitializationMs = performance.now() - webInitStart;
const inputTensor = new ort.Tensor('float32', canonical.tensor, [1, 3, 640, 640]);
const inferWeb = async () => {
  const outputs = await webSession.run({ images: inputTensor });
  if (!outputs.output0) throw new Error('Web output0 missing');
  assertFiniteTensor(outputs.output0, OUTPUT_SHAPE);
  return outputs.output0;
};
let start = performance.now();
let webTensor = await inferWeb();
const webColdMs = performance.now() - start;
for (let index = 0; index < 5; index += 1) webTensor = await inferWeb();
const webSamples = [];
for (let index = 0; index < 30; index += 1) { start = performance.now(); webTensor = await inferWeb(); webSamples.push(performance.now() - start); }
const webPeakRssBytes = await peakRssBytes();
await webSession.release();

const nativeReportPath = resolve(scratch, 'native-report.json');
const nativeScratchOutputPath = resolve(scratch, 'native-output.f32le.bin');
const cargo = spawnSync('cargo', ['run', '--release', '--offline', '--features', 'native', '--example', 'task1_native_benchmark', '--', modelPath, inputPath, nativeReportPath, nativeScratchOutputPath], { cwd: root, encoding: 'utf8' });
if (cargo.status !== 0) throw new Error(`native benchmark failed:\n${cargo.stdout}\n${cargo.stderr}`);
const native = JSON.parse(await readFile(nativeReportPath, 'utf8'));
const nativeOutputBytes = await readFile(nativeScratchOutputPath);
if (nativeOutputBytes.length !== OUTPUT_ELEMENTS * 4) throw new Error(`native output byte length mismatch: ${nativeOutputBytes.length}`);
const nativeOutput = new Float32Array(OUTPUT_ELEMENTS);
for (let index = 0; index < OUTPUT_ELEMENTS; index += 1) {
  const value = nativeOutputBytes.readFloatLE(index * 4);
  if (!Number.isFinite(value)) throw new Error(`Native output contains non-finite value at index ${index}`);
  nativeOutput[index] = value;
}

const webArtifactPath = resolve(artifactDirectory, 'linux-x86_64-single-target-web.f32le.bin');
const nativeArtifactPath = resolve(artifactDirectory, 'linux-x86_64-single-target-native.f32le.bin');
const webArtifact = await writeF32Artifact(webArtifactPath, webTensor.data);
await copyFile(nativeScratchOutputPath, nativeArtifactPath);
const nativeArtifactBytes = await readFile(nativeArtifactPath);
const nativeArtifact = { bytes: nativeArtifactBytes.length, sha256: sha256(nativeArtifactBytes) };
const outputMetadata = (path, artifact, generationMethod) => ({
  classification: 'raw-tensor-test-evidence',
  productPackaging: false,
  modelSha256: sha256(modelBytes),
  fixtureSha256: fixture.sha256,
  inputSha256: canonicalDigest,
  generationMethod,
  licenseSource: 'single-target fixture: ultralytics/assets@42ef8a125df038dcca49f6216f446fe9112946c1, AGPL-3.0-only; test-and-evidence-only; RimeCut product packaging prohibited',
  logicalRole: OUTPUT_ROLE,
  runtimeName: OUTPUT_NAME,
  shape: OUTPUT_SHAPE,
  dtype: OUTPUT_DTYPE,
  endianness: 'little-endian',
  elements: OUTPUT_ELEMENTS,
  finiteCount: OUTPUT_ELEMENTS,
  artifact: { path: relative(root, path), ...artifact },
});
const webOutput = outputMetadata(webArtifactPath, webArtifact, 'onnxruntime-web@1.27.0 WASM single-thread inference');
const nativeOutputMetadata = outputMetadata(nativeArtifactPath, nativeArtifact, 'rimeflow-onnx-base::NativeOrtBackend CPU EP inference');
const numericalComparison = computeNumericalComparison(webTensor.data, nativeOutput);
if (!numericalComparison.passed) throw new Error(`Native/Web raw comparison failed at ${numericalComparison.toleranceMismatchCount} elements`);

const webPostprocessPath = resolve(scratch, 'web-production-postprocess.json');
const nativePostprocessPath = resolve(scratch, 'native-production-postprocess.json');
runProductionPostprocess(operatorRoot, webArtifactPath, fixture.width, fixture.height, webPostprocessPath);
runProductionPostprocess(operatorRoot, nativeArtifactPath, fixture.width, fixture.height, nativePostprocessPath);
const webDetections = JSON.parse(await readFile(webPostprocessPath, 'utf8'));
const nativeDetections = JSON.parse(await readFile(nativePostprocessPath, 'utf8'));
const tolerances = {
  classIdExact: webReference.tolerances.classIdExact,
  confidenceAbsolute: webReference.tolerances.confidenceAbsolute,
  boxIouMinimum: webReference.tolerances.boxIouMinimum,
};
const detectionComparison = compareDetections(webDetections, nativeDetections, tolerances);
if (!detectionComparison.passed) throw new Error('Native/Web production postprocess comparison failed');

const ortArchives = await findFiles(resolve(homedir(), '.cache/ort.pyke.io'), 'libonnxruntime.a');
const nativeRuntimeArtifacts = [];
for (const path of ortArchives) {
  const bytes = await readFile(path);
  nativeRuntimeArtifacts.push({ cachePath: relative(homedir(), path), bytes: bytes.length, sha256: sha256(bytes) });
}
const nativeBinaryPath = resolve(root, 'target/release/examples/task1_native_benchmark');
const nativeBinary = await readFile(nativeBinaryPath);
const webPackagePath = resolve(operatorRoot, 'evidence/tooling/web/node_modules/onnxruntime-web');
const harness = {
  collectorSha256: sha256(await readFile(resolve(root, 'evidence/scripts/collect_linux_baseline.mjs'))),
  numericalValidatorSha256: sha256(await readFile(resolve(root, 'evidence/scripts/performance_validation.mjs'))),
  nativeBenchmarkSha256: sha256(await readFile(resolve(root, 'examples/task1_native_benchmark.rs'))),
  operatorPostprocessSha256: sha256(await readFile(resolve(operatorRoot, 'src/postprocess.rs'))),
  operatorPreprocessSha256: sha256(await readFile(resolve(operatorRoot, 'evidence/scripts/preprocess_contract.mjs'))),
};
const blockedProductMetric = (owner, reason, command, includeScope, excludeScope) => ({
  state: 'blocked', owner, bytes: null, path: null, sha256: null,
  includeScope,
  excludeScope,
  measurementCommand: command, reason,
});
const report = {
  schemaVersion: 2,
  source: {
    baseEvidenceParentCommit,
    operatorEvidenceParentCommit,
    evidenceHarness: harness,
    modelSha256: sha256(modelBytes),
    fixtureId: fixture.id,
    fixtureSha256: fixture.sha256,
    fixtureGeometry: { width: fixture.width, height: fixture.height },
    canonicalInputSha256Float32Le: canonicalDigest,
  },
  host: { os: process.platform, arch: process.arch, node: process.version, cpu: String(spawnSync('sh', ['-c', 'lscpu | sed -n "s/^Model name:[[:space:]]*//p"'], { encoding: 'utf8' }).stdout).trim(), kernel: String(spawnSync('uname', ['-r'], { encoding: 'utf8' }).stdout).trim() },
  method: { warmupRuns: 5, sampleRuns: 30, inputShape: [1, 3, 640, 640], inputDtype: 'float32', byteOrder: 'little-endian', comparisonRule: 'abs(native-web) <= 1e-5 + 1e-4 * abs(web)', relativeDifferenceNearZeroThreshold: 1.0e-6 },
  webWasm: {
    runtime: 'onnxruntime-web@1.27.0', executionProvider: 'wasm', threads: 1,
    metrics: { initializationMs: round(webInitializationMs), coldInferenceMs: round(webColdMs), warmInferenceMs: { p50: round(percentile(webSamples, 0.50)), p95: round(percentile(webSamples, 0.95)), samples: webSamples.map((value) => round(value)) }, peakProcessRssBytes: webPeakRssBytes },
    output: webOutput,
  },
  legacyNativeOrt: {
    ...native,
    adapter: { ...native.adapter, classification: 'wgpu-preprocess-adapter-not-ort-execution-provider' },
    metrics: { ...native.metrics, initializationExcludes: 'wgpuInitializationMs' },
    output: nativeOutputMetadata,
  },
  numericalComparison,
  postprocessComparison: { implementation: 'rimeflow-yolov8n/src/postprocess.rs::decode_yolo_output+nms', tolerances, webDetections, nativeDetections, comparison: detectionComparison },
  packageSizeMetrics: {
    backendRuntimeArtifact: blockedProductMetric(
      'RimeCut product build',
      '当前任务阶段尚无可证明进入最终产品的锁定 backend runtime/model/plugin 制品集合。',
      '在同一 build-once 产品 job 中枚举并哈希实际打包的 runtime/model/plugin 文件',
      '实际进入同平台同配置产品安装包的 runtime/model/plugin 相关制品字节',
      'node_modules 目录、构建缓存、静态 archive、benchmark binary、源目录快照和未进入安装包的文件',
    ),
    finalPackage: blockedProductMetric(
      'RimeCut',
      '没有本轮同平台同配置的最终安装包。',
      'stat --format=%s <final-package> && sha256sum <final-package>',
      '同平台、同架构、同构建配置和同签名阶段的完整 candidate 最终可安装制品',
      '中间构建目录、未打包文件、缓存、调试符号和其他非最终安装制品',
    ),
    legacyFinalPackage: blockedProductMetric(
      'RimeCut',
      '没有锁定 Legacy 最终安装包及其 SHA-256。',
      'stat --format=%s <legacy-final-package> && sha256sum <legacy-final-package>',
      '与 candidate 同平台、同架构、同构建配置和同签名阶段的完整锁定 Legacy 最终可安装制品',
      '中间构建目录、未打包文件、缓存、调试符号和其他非最终安装制品',
    ),
    finalPackageGrowth: { state: 'blocked', bytes: null, ratio: null, formula: { bytes: 'candidateFinalPackageBytes - legacyFinalPackageBytes', ratio: 'candidateFinalPackageBytes / legacyFinalPackageBytes' }, reason: 'candidate 与 Legacy 最终安装包均缺失；missingMetricPolicy=fail。' },
  },
  componentObservations: {
    comparableToProductPackage: false,
    webNpmPackageDirectory: { path: relative(root, webPackagePath), bytes: await directoryBytes(webPackagePath), measurementCommand: `du -sb ${relative(root, webPackagePath)}`, reason: '完整 npm package 目录包含未必进入产品的文件，不能作为 backendRuntimeArtifactBytes 或 finalPackageBytes。' },
    nativeOrtStaticArchives: nativeRuntimeArtifacts,
    benchmarkBinary: { path: relative(root, nativeBinaryPath), bytes: nativeBinary.length, sha256: sha256(nativeBinary), reason: '采集 harness，不是最终安装包。' },
  },
  comparability: {
    scope: 'same Linux x86_64 host backend microbenchmark', sameDevice: true, crossBackendProviderEqual: false,
    providerIdentity: {
      web: { provider: 'wasm', roundProviders: ['wasm', 'wasm'], stableAcrossRounds: true },
      native: { provider: 'Cpu', roundProviders: ['Cpu', 'Cpu'], stableAcrossRounds: true },
    },
    sameModelDigest: true, sameFixtureDigest: true, sameInputDigest: true,
    webExecution: 'WASM single-thread', nativeResolvedProvider: 'CPU', lifecycleEquivalent: false,
    lifecycleNote: 'Native initializationMs excludes separately recorded wgpuInitializationMs; the Web and Native initialization lifecycles are not claimed to be identical.',
  },
  productMetrics: {
    owner: 'RimeCut', evaluationState: 'not-evaluated-by-base',
    metrics: ['backendRuntimeArtifactBytes', 'finalPackageBytes', 'finalPackageGrowthBytes', 'finalPackageGrowthRatio', 'rollbackPackage'],
    baseClaim: 'none',
  },
  limitation: '这是同一 Linux x86_64 主机上的固定输入 backend microbenchmark；不代表产品端到端延迟或最终包体积，也不替代其他平台同设备比较。',
};
const temporaryOutputPath = `${outputPath}.tmp-${process.pid}`;
await writeFile(temporaryOutputPath, `${JSON.stringify(report, null, 2)}\n`);
await rename(temporaryOutputPath, outputPath);
console.log(JSON.stringify({ output: relative(root, outputPath), numericalComparison: report.numericalComparison, postprocessComparison: report.postprocessComparison.comparison }));
