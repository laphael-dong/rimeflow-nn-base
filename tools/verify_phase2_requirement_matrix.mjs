// RFB-BASE-TRACE-001
import { readFileSync, readdirSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = process.cwd();
const matrixPath = join(root, 'contract-tests/phase2-base-requirement-test-matrix.json');
const matrix = JSON.parse(readFileSync(matrixPath, 'utf8'));
const idPattern = /^RFB-BASE-[A-Z0-9-]+$/;

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

assert(matrix.schemaVersion === 1, 'schemaVersion must be 1');
assert(matrix.owner === 'rimeflow-nn-base', 'owner must be rimeflow-nn-base');
assert(matrix.phase === 2 && matrix.status === 'test-first-red', 'phase/status must describe Phase 2 red tests');
assert(Array.isArray(matrix.entries) && matrix.entries.length > 0, 'entries must be non-empty');

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
console.log(`Phase 2 Base matrix valid: ${matrix.entries.length} entries, ${matrixIds.size} unique IDs.`);
