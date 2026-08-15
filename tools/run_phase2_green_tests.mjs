import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const matrixPath = join(root, 'contract-tests/phase2-base-requirement-test-matrix.json');
const reportRelativePath = 'reports/phase2-base-runtime-implementation-report.json';
const reportPath = join(root, reportRelativePath);
const matrix = JSON.parse(readFileSync(matrixPath, 'utf8'));
const commitSha = process.argv[2];
const commitPattern = /^[0-9a-f]{40}$/;

function assert(condition, message) {
  if (!condition) throw new Error(`Phase 2 Base green-test runner: ${message}`);
}

assert(commitPattern.test(commitSha ?? ''), 'first argument must be the implementation commit SHA');
const lockedModelPath = process.env.RIMEFLOW_YOLOV8N_MODEL;
assert(lockedModelPath, 'RIMEFLOW_YOLOV8N_MODEL must name the locked validation model');
const lockedModelBytes = readFileSync(lockedModelPath);
const lockedModelSha256 = createHash('sha256').update(lockedModelBytes).digest('hex');
assert(statSync(lockedModelPath).size === 12_851_098, 'locked model byte length drifted');
assert(lockedModelSha256 === '9e7e3921595672c4b97e78f78bf5604d86ffc117773da49f142d1047109d07ad', 'locked model SHA-256 drifted');

const head = spawnSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' });
assert(head.status === 0, `cannot read HEAD: ${head.stderr}`);
assert(head.stdout.trim() === commitSha, `requested commit ${commitSha} is not current HEAD ${head.stdout.trim()}`);

const suites = [
  { file: 'tests/phase2_manifest_red.rs', command: ['test', '--test', 'phase2_manifest_red'], expectedTests: 5 },
  { file: 'tests/phase2_lifecycle_red.rs', command: ['test', '--test', 'phase2_lifecycle_red'], expectedTests: 8 },
  { file: 'tests/phase2_adapter_conformance_red.rs', command: ['test', '--test', 'phase2_adapter_conformance_red'], expectedTests: 2 },
  { file: 'tests/manifest_contract.rs', command: ['test', '--test', 'manifest_contract'], expectedTests: 5 },
  { file: 'tests/runtime_contract.rs', command: ['test', '--test', 'runtime_contract'], expectedTests: 6 },
];

const suiteResults = suites.map((suite) => {
  const execution = spawnSync('cargo', suite.command, { cwd: root, encoding: 'utf8' });
  const output = `${execution.stdout ?? ''}${execution.stderr ?? ''}`;
  assert(execution.error === undefined, `${suite.file} could not execute: ${execution.error}`);
  assert(execution.status === 0, `${suite.file} failed with exit ${execution.status}:\n${output}`);
  assert(output.includes(`running ${suite.expectedTests} tests`), `${suite.file} did not run ${suite.expectedTests} tests`);
  assert(output.includes('test result: ok.'), `${suite.file} did not report a green test result`);
  return {
    file: suite.file,
    command: `cargo ${suite.command.join(' ')}`,
    exitCode: execution.status,
    passedTests: suite.expectedTests,
    status: 'passed',
  };
});

const schemaExecution = spawnSync('node', ['tools/verify_model_manifest_fixtures.mjs'], {
  cwd: root,
  encoding: 'utf8',
});
const schemaOutput = `${schemaExecution.stdout ?? ''}${schemaExecution.stderr ?? ''}`;
assert(schemaExecution.error === undefined, `schema fixture verification could not execute: ${schemaExecution.error}`);
assert(schemaExecution.status === 0, `schema fixture verification failed with exit ${schemaExecution.status}:\n${schemaOutput}`);
assert(schemaOutput.includes('5 fixtures'), 'schema fixture verification did not cover all fixed fixtures');

const smokeCommand = ['test', '--features', 'native', '--test', 'legacy_ort_smoke', '--', '--ignored'];
const smokeExecution = spawnSync('cargo', smokeCommand, {
  cwd: root,
  encoding: 'utf8',
  env: process.env,
});
const smokeOutput = `${smokeExecution.stdout ?? ''}${smokeExecution.stderr ?? ''}`;
assert(smokeExecution.error === undefined, `Legacy ORT smoke could not execute: ${smokeExecution.error}`);
assert(smokeExecution.status === 0, `Legacy ORT smoke failed with exit ${smokeExecution.status}:\n${smokeOutput}`);
assert(smokeOutput.includes('running 1 test'), 'Legacy ORT smoke did not run exactly one real-model test');
assert(smokeOutput.includes('test result: ok.'), 'Legacy ORT smoke did not report a green result');

const traceExecution = spawnSync('node', ['tools/verify_phase2_requirement_matrix.mjs'], {
  cwd: root,
  encoding: 'utf8',
});
const traceOutput = `${traceExecution.stdout ?? ''}${traceExecution.stderr ?? ''}`;
assert(traceExecution.error === undefined, `matrix verification could not execute: ${traceExecution.error}`);
assert(traceExecution.status === 0, `matrix verification failed with exit ${traceExecution.status}:\n${traceOutput}`);
assert(traceOutput.includes('17 entries, 17 unique IDs'), 'matrix verification did not cover all 17 IDs');

const results = matrix.entries.map((entry) => {
  const suite = entry.test.file.startsWith('tests/')
    ? suiteResults.find((candidate) => candidate.file === entry.test.file)
    : null;
  const trace = entry.test.file === 'tools/verify_phase2_requirement_matrix.mjs';
  assert(suite || trace, `${entry.id} has no executed green suite`);
  return {
    id: entry.id,
    result: 'passed',
    commitSha,
    test: entry.test,
    executedCommand: suite?.command ?? 'node tools/verify_phase2_requirement_matrix.mjs',
    exitCode: suite?.exitCode ?? traceExecution.status,
  };
});

const report = {
  schemaVersion: 1,
  owner: 'rimeflow-nn-base',
  phase: 2,
  status: 'passed',
  commitSha,
  modelIdentity: {
    repository: 'https://github.com/laphael-dong/rimeflow-nn-validation.git',
    commit: 'c90d3957fbbd04b3f0b29eff7bc873b70eed4400',
    tree: '341d8b00fb5d4d9afeac856418950c1faa408b2e',
    path: 'models/yolov8n.onnx',
    blob: '22f19afe710dfa942b3e644c4e5a7ac5c42ac403',
    bytes: lockedModelBytes.length,
    sha256: lockedModelSha256,
  },
  results,
  suites: [
    ...suiteResults,
    {
      file: 'tests/legacy_ort_smoke.rs',
      command: `RIMEFLOW_YOLOV8N_MODEL=<locked-model> cargo ${smokeCommand.join(' ')}`,
      exitCode: smokeExecution.status,
      passedTests: 1,
      status: 'passed',
    },
    {
      file: 'tools/verify_model_manifest_fixtures.mjs',
      command: 'node tools/verify_model_manifest_fixtures.mjs',
      exitCode: schemaExecution.status,
      passedTests: 5,
      status: 'passed',
    },
    {
      file: 'tools/verify_phase2_requirement_matrix.mjs',
      command: 'node tools/verify_phase2_requirement_matrix.mjs',
      exitCode: traceExecution.status,
      passedTests: 2,
      status: 'passed',
    },
  ],
};
mkdirSync(join(root, 'reports'), { recursive: true });
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(`Phase 2 Base green tests verified: 17/17 IDs. Report: ${reportRelativePath}`);
