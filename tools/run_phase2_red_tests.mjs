import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';

const root = process.cwd();
const cases = [
  {
    name: 'phase2_manifest_red',
    expectedTests: 5,
    operations: ['not_implemented:manifest.schema_validate', 'not_implemented:manifest.semantic_validate'],
  },
  {
    name: 'phase2_lifecycle_red',
    expectedTests: 8,
    operations: ['not_implemented:runtime.native_initialize', 'not_implemented:runtime.web_initialize', 'not_implemented:runtime.release', 'not_implemented:runtime.infer'],
  },
  {
    name: 'phase2_adapter_conformance_red',
    expectedTests: 2,
    operations: ['not_implemented:adapter.conformance'],
  },
];

function assert(condition, message) {
  if (!condition) throw new Error(`Phase 2 Base red-test runner: ${message}`);
}

const results = cases.map((testCase) => {
  const command = ['test', '--test', testCase.name, '--', '--nocapture'];
  const execution = spawnSync('cargo', command, { cwd: root, encoding: 'utf8' });
  const output = `${execution.stdout ?? ''}${execution.stderr ?? ''}`;
  assert(execution.error === undefined, `${testCase.name} could not execute: ${execution.error}`);
  assert(execution.status === 101, `${testCase.name} must fail with Cargo test status 101, observed ${execution.status}`);
  assert(output.includes(`running ${testCase.expectedTests} tests`), `${testCase.name} did not run ${testCase.expectedTests} tests`);
  assert(output.includes('test result: FAILED.'), `${testCase.name} did not fail through test assertions`);
  for (const operation of testCase.operations) {
    assert(output.includes(operation), `${testCase.name} did not fail at ${operation}`);
  }
  return {
    test: testCase.name,
    command: `cargo ${command.join(' ')}`,
    exitCode: execution.status,
    expectedTests: testCase.expectedTests,
    assertionFailures: testCase.operations,
  };
});

const report = {
  schemaVersion: 1,
  owner: 'rimeflow-nn-base',
  phase: 2,
  status: 'expected-red',
  results,
};
const reportPath = join(root, 'target/phase2-base-red-test-report.json');
mkdirSync(join(root, 'target'), { recursive: true });
writeFileSync(reportPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(`Phase 2 Base expected red tests verified: ${results.length} suites. Report: ${reportPath}`);
