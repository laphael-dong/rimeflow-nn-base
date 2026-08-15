// RFB-BASE-TRACE-001
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = process.cwd();
const matrixPath = join(root, 'contract-tests/phase2-base-requirement-test-matrix.json');
const matrix = JSON.parse(readFileSync(matrixPath, 'utf8'));
const idPattern = /^RFB-BASE-[A-Z0-9-]+$/;
const commitPattern = /^[0-9a-f]{40}$/;

function assert(condition, message) {
  if (!condition) throw new Error(`Phase 2 Base matrix: ${message}`);
}

function readTrackedFile(path) {
  try {
    return readFileSync(join(root, path), 'utf8');
  } catch (error) {
    throw new Error(`Phase 2 Base matrix: cannot read ${path}: ${error.message}`);
  }
}

function collectRustTestFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return collectRustTestFiles(path);
    return entry.isFile() && entry.name.endsWith('.rs') ? [path] : [];
  });
}

function validateCompletionEvidence(entry, defaultStatus) {
  const status = entry.status ?? defaultStatus;
  assert(status === 'test-first-red' || status === 'complete', `${entry.id} has unsupported status ${status}`);
  if (status !== 'complete') return;

  assert(entry.expected.startsWith('green:'), `${entry.id} complete status requires a green expected outcome`);
  const evidence = entry.completionEvidence;
  assert(evidence && typeof evidence === 'object', `${entry.id} complete status requires completionEvidence`);
  assert(evidence.result === 'passed', `${entry.id} complete status requires a passed executable result`);
  assert(evidence.reportPath === entry.reportPath, `${entry.id} completion evidence must bind its reportPath`);
  assert(commitPattern.test(evidence.commitSha), `${entry.id} completion evidence must bind a 40-character commit SHA`);
}

function assertRejected(action, expectedMessage) {
  try {
    action();
  } catch (error) {
    assert(error instanceof Error && error.message.includes(expectedMessage), `completion guard failed unexpectedly: ${error}`);
    return;
  }
  throw new Error(`Phase 2 Base matrix: completion guard accepted invalid state: ${expectedMessage}`);
}

// RFB-BASE-TRACE-002: marking a requirement complete is invalid without commit-bound passing evidence.
assertRejected(
  () => validateCompletionEvidence({ id: 'completion-probe', status: 'complete', expected: 'red: pending', reportPath: 'probe.json' }, matrix.status),
  'complete status requires a green expected outcome',
);
assertRejected(
  () => validateCompletionEvidence({ id: 'completion-probe', status: 'complete', expected: 'green: passed', reportPath: 'probe.json' }, matrix.status),
  'complete status requires completionEvidence',
);
validateCompletionEvidence(
  {
    id: 'completion-probe',
    status: 'complete',
    expected: 'green: passed',
    reportPath: 'probe.json',
    completionEvidence: { result: 'passed', reportPath: 'probe.json', commitSha: 'a'.repeat(40) },
  },
  matrix.status,
);

assert(matrix.schemaVersion === 1, 'schemaVersion must be 1');
assert(matrix.owner === 'rimeflow-nn-base', 'owner must be rimeflow-nn-base');
assert(matrix.phase === 2, 'phase must be 2');
assert(matrix.status === 'test-first-red' || matrix.status === 'complete', 'matrix status must be test-first-red or complete');
assert(Array.isArray(matrix.entries) && matrix.entries.length > 0, 'entries must be non-empty');

let completionReport = null;
if (matrix.status === 'complete') {
  const reportPaths = new Set(matrix.entries.map((entry) => entry.reportPath));
  assert(reportPaths.size === 1, 'complete matrix entries must share one reportPath');
  const [reportPath] = reportPaths;
  assert(existsSync(join(root, reportPath)), `complete report does not exist: ${reportPath}`);
  completionReport = JSON.parse(readTrackedFile(reportPath));
  assert(completionReport.status === 'passed', 'complete report status must be passed');
  assert(commitPattern.test(completionReport.commitSha), 'complete report must bind a 40-character commit SHA');
  assert(Array.isArray(completionReport.results), 'complete report results must be an array');
}

const matrixIds = new Set();
const listedTestFiles = new Set();
for (const entry of matrix.entries) {
  assert(idPattern.test(entry.id), `invalid test ID ${entry.id}`);
  assert(!matrixIds.has(entry.id), `duplicate test ID ${entry.id}`);
  matrixIds.add(entry.id);
  for (const field of ['spec', 'requirement', 'scenario', 'reportPath', 'expected']) {
    assert(typeof entry[field] === 'string' && entry[field].trim() !== '', `${entry.id} has no ${field}`);
    assert(!entry[field].includes('TBD'), `${entry.id} has TBD in ${field}`);
  }
  assert(entry.test && typeof entry.test === 'object', `${entry.id} has no test descriptor`);
  assert(typeof entry.test.file === 'string' && typeof entry.test.name === 'string', `${entry.id} has incomplete test descriptor`);
  assert(typeof entry.test.command === 'string' && entry.test.command.trim() !== '', `${entry.id} has no command`);
  assert(entry.test.command.startsWith('cargo test ') || entry.test.command.startsWith('node '), `${entry.id} has an unsupported command`);
  assert(entry.expected.includes('not_implemented') || entry.expected.startsWith('green:'), `${entry.id} must declare an expected red or green outcome`);
  validateCompletionEvidence(entry, matrix.status);
  if (completionReport) {
    const evidence = completionReport.results.find((result) => result.id === entry.id);
    assert(evidence, `${entry.id} has no result in ${entry.reportPath}`);
    assert(evidence.result === 'passed', `${entry.id} report result is not passed`);
    assert(evidence.test.file === entry.test.file, `${entry.id} report test file drifted`);
    assert(evidence.test.name === entry.test.name, `${entry.id} report test name drifted`);
    assert(evidence.test.command === entry.test.command, `${entry.id} report test command drifted`);
    assert(evidence.commitSha === entry.completionEvidence.commitSha, `${entry.id} report commit drifted`);
    assert(evidence.commitSha === completionReport.commitSha, `${entry.id} does not bind the report commit`);
  }
  const source = readTrackedFile(entry.test.file);
  assert(source.includes(entry.id), `${entry.id} is not declared in ${entry.test.file}`);
  listedTestFiles.add(entry.test.file);
}

const declaredIds = new Set();
for (const file of listedTestFiles) {
  const source = readTrackedFile(file);
  for (const id of source.matchAll(/RFB-BASE-[A-Z0-9-]+/g)) declaredIds.add(id[0]);
}

for (const absolutePath of collectRustTestFiles(join(root, 'tests'))) {
  const file = relative(root, absolutePath);
  const source = readFileSync(absolutePath, 'utf8');
  for (const id of source.matchAll(/RFB-BASE-[A-Z0-9-]+/g)) {
    assert(matrixIds.has(id[0]), `orphan test ID ${id[0]} in ${file}`);
  }
}

for (const id of matrixIds) assert(declaredIds.has(id), `matrix ID ${id} has no source declaration`);
if (completionReport) {
  const reportIds = new Set(completionReport.results.map((result) => result.id));
  assert(reportIds.size === matrixIds.size, 'complete report has duplicate or extra result IDs');
  for (const id of reportIds) assert(matrixIds.has(id), `complete report has unknown ID ${id}`);
}
console.log(`Phase 2 Base matrix valid: ${matrix.entries.length} entries, ${matrixIds.size} unique IDs.`);
