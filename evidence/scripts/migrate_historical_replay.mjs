import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const root = resolve(import.meta.dirname, '../..');
const replayPath = resolve(root, 'evidence/replay/task1-replay.json');
const replay = JSON.parse(await readFile(replayPath, 'utf8'));
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));
const historicalStepsDigest = createHash('sha256').update(JSON.stringify(replay.steps)).digest('hex');
if (historicalStepsDigest !== '90181c2efc5b3155572c1d63718e37275f440c0e13d2df2239bccf88a0837b41') throw new Error('historical replay steps drift; refusing to rewrite history');
const outputPaths = [
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
const outputs = [];
for (const path of outputPaths) {
  const bytes = await readFile(resolve(root, path));
  outputs.push({ path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
}
const migrated = {
  schemaVersion: 3,
  repository: 'rimeflow-nn-base',
  historicalReplay: publication.historicalReplay,
  measurementIdentity: publication.measurementIdentity,
  operatorInputPublication: publication.operatorInputPublication,
  basePublicationState: publication.basePublicationState,
  runner: replay.runner,
  tools: replay.tools,
  immutableLogEvidence: {
    kind: 'embedded-historical-manifest',
    path: 'evidence/replay/task1-replay.json',
    formalCiState: publication.requiredCiState
  },
  steps: replay.steps,
  twoRoundPerformanceOutputDigestsEqual: replay.twoRoundPerformanceOutputDigestsEqual,
  outputs,
  ordinaryReplayMutatesTrackedEvidence: false,
  ordinaryReplayValidationChain: ['main-validator', 'performance-negative', 'publication-schema-negative', 'replay-negative', 'main-security-negative', 'official-trust-negative'],
  task1_6ExecutableClosure: false,
  task1_6BlockedReason: 'Candidate adapter、RimeCut 产品指标、正式 CI 与远端 publication 尚未完成；本地历史基线迁移不代表任务 1.6 完成。',
  task1_7OwnershipReplayComplete: false
};
await writeFile(replayPath, `${JSON.stringify(migrated, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/replay/task1-replay.json', historicalReplayHead: publication.historicalReplay.head, outputCount: outputs.length }));
