import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { blockedStep, runRepeatedStep, runnerIdentity, sha256, toolVersion } from './task1_replay_execution.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const outputPath = resolve(root, 'evidence/replay/task1-replay.json');
if (process.env.RIMEFLOW_RECORD_REPLAY !== '1') throw new Error('record mode requires RIMEFLOW_RECORD_REPLAY=1; ordinary replay does not modify tracked evidence');
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));
const outputs = [
  'evidence/schemas/platform-matrix.schema.json',
  'evidence/platform/platform-matrix.json',
  'evidence/platform/runner-inventory.json',
  'evidence/reports/local-environment.json',
  'evidence/performance/backend-thresholds.json',
  'evidence/performance/linux-x86_64-capture.json',
  'evidence/performance/linux-x86_64-baseline.json',
  'evidence/performance/artifacts/linux-x86_64-single-target-web.f32le.bin',
  'evidence/performance/artifacts/linux-x86_64-single-target-native.f32le.bin',
  'evidence/publication/task1-publication.json',
];
const capture = JSON.parse(await readFile(resolve(root, 'evidence/performance/linux-x86_64-capture.json'), 'utf8'));
const operatorRoot = resolve(root, process.env.RIMEFLOW_OPERATOR_ROOT ?? '../rimeflow-yolov8n');
const replayCapturePaths = ['.evidence/task1-linux-baseline/replay-capture-round-1.json', '.evidence/task1-linux-baseline/replay-capture-round-2.json'];
const steps = [];
steps.push(await runRepeatedStep({
  root,
  id: 'platform-matrix',
  command: 'RIMEFLOW_OPERATOR_ROOT=../rimeflow-yolov8n node evidence/scripts/generate_platform_evidence.mjs',
  executable: 'node',
  args: ['evidence/scripts/generate_platform_evidence.mjs'],
  env: { RIMEFLOW_OPERATOR_ROOT: operatorRoot },
  inputPaths: ['evidence/schemas/platform-matrix.schema.json'],
  outputPaths: ['evidence/platform/platform-matrix.json', 'evidence/platform/runner-inventory.json', 'evidence/reports/local-environment.json'],
}));
for (let index = 0; index < replayCapturePaths.length; index += 1) {
  steps.push(await runRepeatedStep({
    root,
    id: `linux-performance-capture-round-${index + 1}`,
    command: `RIMEFLOW_OPERATOR_ROOT=../rimeflow-yolov8n node evidence/scripts/collect_linux_baseline.mjs ${replayCapturePaths[index]}`,
    executable: 'node',
    args: ['evidence/scripts/collect_linux_baseline.mjs', replayCapturePaths[index]],
    env: { RIMEFLOW_OPERATOR_ROOT: operatorRoot },
    inputPaths: ['evidence/performance/backend-thresholds.json'],
    outputPaths: [replayCapturePaths[index]],
    runs: 1,
  }));
}
const replayCaptures = await Promise.all(replayCapturePaths.map(async (path) => JSON.parse(await readFile(resolve(root, path), 'utf8'))));
const replayOutputDigestsEqual = replayCaptures[0].webWasm.output.artifact.sha256 === replayCaptures[1].webWasm.output.artifact.sha256
  && replayCaptures[0].legacyNativeOrt.output.artifact.sha256 === replayCaptures[1].legacyNativeOrt.output.artifact.sha256;
steps.push(await runRepeatedStep({
  root,
  id: 'performance-baseline',
  command: 'node evidence/scripts/generate_performance_baseline.mjs',
  executable: 'node',
  args: ['evidence/scripts/generate_performance_baseline.mjs'],
  inputPaths: ['evidence/performance/linux-x86_64-capture.json'],
  outputPaths: ['evidence/performance/linux-x86_64-baseline.json'],
}));
steps.push(blockedStep('product-package-comparison', 'owned by RimeCut build-once product job', 'Legacy 与 candidate 最终安装包均不存在；相对包体积指标按 missingMetricPolicy=fail blocked。'));
const artifacts = [];
for (const path of outputs) {
  const bytes = await readFile(resolve(root, path));
  artifacts.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
}
const manifest = {
  schemaVersion: 2,
  repository: 'rimeflow-nn-base',
  historicalReplay: publication.historicalReplay,
  measurementIdentity: publication.measurementIdentity,
  operatorInputPublication: publication.operatorInputPublication,
  basePublicationState: publication.basePublicationState,
  runner: runnerIdentity(),
  tools: { node: toolVersion(root, 'node', ['--version']), cargo: toolVersion(root, 'cargo', ['--version']), onnxruntimeWeb: capture.webWasm.runtime, nativeOrt: capture.legacyNativeOrt.runtime.ortCrate },
  immutableLogEvidence: { kind: 'embedded-historical-manifest', path: 'evidence/replay/task1-replay.json', formalCiState: publication.requiredCiState },
  steps,
  twoRoundPerformanceOutputDigestsEqual: replayOutputDigestsEqual,
  outputs: artifacts,
  task1_6ExecutableClosure: false,
  task1_6BlockedReason: 'Candidate adapter、RimeCut 产品指标、正式 CI 与远端 publication 尚未完成；本地历史基线迁移不代表任务 1.6 完成。',
  ordinaryReplayMutatesTrackedEvidence: false,
  ordinaryReplayValidationChain: ['main-validator', 'performance-negative', 'publication-schema-negative', 'replay-negative', 'main-security-negative', 'official-trust-negative'],
  task1_7OwnershipReplayComplete: false,
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(manifest, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', outputCount: artifacts.length }));
