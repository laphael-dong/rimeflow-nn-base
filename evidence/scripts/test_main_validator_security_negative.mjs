import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { validateEvidence } from './validate_evidence.mjs';

const root = resolve(import.meta.dirname, '../..');
const operatorBundle = process.env.RIMEFLOW_OPERATOR_BUNDLE;
if (!operatorBundle?.startsWith('/')) throw new Error('main security negative suite requires absolute RIMEFLOW_OPERATOR_BUNDLE');
const scratch = await mkdtemp(join(tmpdir(), 'rimeflow-task16-fix02-main-negative-'));
const cases = [];
const load = async (base, path) => JSON.parse(await readFile(resolve(base, path), 'utf8'));
const save = async (base, path, value) => writeFile(resolve(base, path), `${JSON.stringify(value, null, 2)}\n`);

const rejectMain = async (name, mutate, pattern) => {
  const candidate = resolve(scratch, `case-${String(cases.length + 1).padStart(2, '0')}`);
  await cp(resolve(root, 'evidence'), resolve(candidate, 'evidence'), { recursive: true });
  await mutate(candidate);
  try {
    await validateEvidence({ root: candidate, operatorBundleRoot: operatorBundle, operatorRoot: resolve(operatorBundle, 'source'), scratchRoot: resolve(candidate, 'scratch') });
  } catch (error) {
    if (!pattern.test(error.message)) throw new Error(`${name}: unexpected rejection: ${error.message}`);
    cases.push(name);
    return;
  }
  throw new Error(`main-path security case unexpectedly passed: ${name}`);
};

const validApproval = {
  approver: 'reviewer', submitter: 'author', platform: 'linux-x86_64-ort-cpu', metric: 'warmInferenceMs',
  candidateCommit: '1'.repeat(40), modelSha256: '2'.repeat(64), artifactDigest: '3'.repeat(64),
  runtimeName: 'onnxruntime', runtimeVersion: '1.24.3', observed: 101, threshold: 100,
  reason: 'Accepted bounded regression', createdAt: '2026-08-01T00:00:00.000Z', expiresAt: '2026-08-13T00:00:00.000Z',
};

try {
  await rejectMain('combined Schema matrix replay identity and future approval bypass', async (base) => {
    await writeFile(resolve(base, 'evidence/schemas/platform-matrix.schema.json'), '{}\n');
    const matrix = await load(base, 'evidence/platform/platform-matrix.json');
    delete matrix.platforms[0].minimumOsVersion;
    matrix.platforms[0].unknown = true;
    await save(base, 'evidence/platform/platform-matrix.json', matrix);
    const replay = await load(base, 'evidence/replay/task1-replay.json');
    replay.outputs = [];
    replay.runner.owner = 'fabricated';
    replay.immutableLogEvidence.formalCiState = 'passed';
    replay.repositoryHeadAtReplay = '0'.repeat(40);
    await save(base, 'evidence/replay/task1-replay.json', replay);
    const thresholds = await load(base, 'evidence/performance/backend-thresholds.json');
    thresholds.approvalRule.currentApprovals = [{ ...validApproval, platform: 'unknown-platform', metric: 'inventedMetric', runtimeName: 'invented-runtime', createdAt: '2026-08-14T00:00:00.000Z', expiresAt: '2026-08-15T00:00:00.000Z' }];
    await save(base, 'evidence/performance/backend-thresholds.json', thresholds);
  }, /frozen platform Schema SHA-256 mismatch/);

  const replayCases = [
    ['empty replay outputs', (x) => { x.outputs = []; }, /output ledger length/],
    ['missing Schema replay output', (x) => { x.outputs.shift(); }, /output ledger length/],
    ['duplicate replay output', (x) => { x.outputs[1].path = x.outputs[0].path; }, /duplicate output path/],
    ['reordered replay outputs', (x) => { [x.outputs[0], x.outputs[1]] = [x.outputs[1], x.outputs[0]]; }, /output tuples/],
    ['runner identity drift', (x) => { x.runner.owner = 'fabricated'; }, /runner identity/],
    ['formal CI overclaim', (x) => { x.immutableLogEvidence.formalCiState = 'passed'; }, /log\/CI identity/],
    ['unknown replay top-level field', (x) => { x.repositoryHeadAtReplay = '0'.repeat(40); }, /top-level fields/],
  ];
  for (const [name, mutate, pattern] of replayCases) await rejectMain(name, async (base) => {
    const replay = await load(base, 'evidence/replay/task1-replay.json');
    mutate(replay);
    await save(base, 'evidence/replay/task1-replay.json', replay);
  }, pattern);
  await rejectMain('tracked replay output is a directory', async (base) => {
    const rawPath = resolve(base, 'evidence/performance/artifacts/linux-x86_64-single-target-web.f32le.bin');
    await rm(rawPath);
    await mkdir(rawPath);
  }, /tracked evidence is not a regular file/);

  const approvalCases = [
    ['otherwise valid tracked approval rejected without request', () => {}],
    ['tracked approval unknown platform', (x) => { x.platform = 'unknown-platform'; }],
    ['tracked approval invented metric', (x) => { x.metric = 'inventedMetric'; }],
    ['tracked approval candidate mismatch', (x) => { x.candidateCommit = '4'.repeat(40); }],
    ['tracked approval model mismatch', (x) => { x.modelSha256 = '5'.repeat(64); }],
    ['tracked approval artifact mismatch', (x) => { x.artifactDigest = '6'.repeat(64); }],
    ['tracked approval runtime name mismatch', (x) => { x.runtimeName = 'invented-runtime'; }],
    ['tracked approval runtime version mismatch', (x) => { x.runtimeVersion = '9.9.9'; }],
    ['tracked approval future creation', (x) => { x.createdAt = '2026-08-14T00:00:00.000Z'; x.expiresAt = '2026-08-15T00:00:00.000Z'; }],
    ['tracked approval extra field', (x) => { x.extra = true; }],
  ];
  for (const [name, mutate] of approvalCases) await rejectMain(name, async (base) => {
    const thresholds = await load(base, 'evidence/performance/backend-thresholds.json');
    const approval = structuredClone(validApproval);
    mutate(approval);
    thresholds.approvalRule.currentApprovals = [approval];
    await save(base, 'evidence/performance/backend-thresholds.json', thresholds);
  }, /tracked approvals must remain empty/);

  console.log(JSON.stringify({ ok: true, policy: 'tracked approvals remain empty until a candidate exception request exists', expiryBoundary: 'expiresAt equal to validation now is structurally valid but rejected as unowned tracked approval', negativeCaseCount: cases.length, negativeCases: cases }));
} finally {
  await rm(scratch, { recursive: true, force: true });
}
