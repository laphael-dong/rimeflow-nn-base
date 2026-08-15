import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const matrixPath = join(root, 'contract-tests/phase2-base-requirement-test-matrix.json');
const reportRelativePath = 'reports/phase2-base-runtime-implementation-report.json';
const reportPath = join(root, reportRelativePath);
const commitSha = process.argv[2];
const commitPattern = /^[0-9a-f]{40}$/;

function assert(condition, message) {
  if (!condition) throw new Error(`Phase 2 Base matrix finalizer: ${message}`);
}

assert(commitPattern.test(commitSha ?? ''), 'first argument must be the implementation commit SHA');
const matrix = JSON.parse(readFileSync(matrixPath, 'utf8'));
const report = JSON.parse(readFileSync(reportPath, 'utf8'));
assert(report.status === 'passed', 'green report status must be passed');
assert(report.commitSha === commitSha, 'green report commit does not match requested commit');
assert(Array.isArray(report.results) && report.results.length === matrix.entries.length, 'green report result count does not match matrix');

matrix.status = 'complete';
for (const entry of matrix.entries) {
  const result = report.results.find((candidate) => candidate.id === entry.id);
  assert(result?.result === 'passed', `${entry.id} has no passing green report result`);
  assert(result.test.file === entry.test.file, `${entry.id} test file drifted`);
  assert(result.test.name === entry.test.name, `${entry.id} test name drifted`);
  assert(result.test.command === entry.test.command, `${entry.id} test command drifted`);
  entry.status = 'complete';
  entry.reportPath = reportRelativePath;
  entry.expected = `green: ${entry.test.name} passed`;
  entry.completionEvidence = {
    result: 'passed',
    reportPath: reportRelativePath,
    commitSha,
  };
}

writeFileSync(matrixPath, `${JSON.stringify(matrix, null, 2)}\n`);
console.log(`Phase 2 Base matrix finalized: ${matrix.entries.length}/17 entries bind ${commitSha}.`);
