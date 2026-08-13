import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

import { validatePerformanceCapture } from './performance_validation.mjs';

const root = resolve(import.meta.dirname, '../..');
const operatorRoot = resolve(process.env.RIMEFLOW_OPERATOR_ROOT ?? resolve(root, '../rimeflow-yolov8n'));
const original = JSON.parse(await readFile(resolve(root, 'evidence/performance/linux-x86_64-capture.json'), 'utf8'));
const temporary = await mkdtemp(join(tmpdir(), 'rimeflow-performance-negative-'));
const cases = [];

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

async function expectFailure(name, expectedMessage, mutate) {
  const capture = structuredClone(original);
  const webBytes = await readFile(resolve(root, original.webWasm.output.artifact.path));
  const nativeBytes = await readFile(resolve(root, original.legacyNativeOrt.output.artifact.path));
  await writeFile(join(temporary, `${name}-web.bin`), webBytes);
  await writeFile(join(temporary, `${name}-native.bin`), nativeBytes);
  capture.webWasm.output.artifact.path = `${name}-web.bin`;
  capture.legacyNativeOrt.output.artifact.path = `${name}-native.bin`;
  capture.webWasm.output.artifact.bytes = webBytes.length;
  capture.legacyNativeOrt.output.artifact.bytes = nativeBytes.length;
  await mutate({ capture, webBytes: Buffer.from(webBytes), nativeBytes: Buffer.from(nativeBytes), temporary });
  try {
    await validatePerformanceCapture(temporary, operatorRoot, capture, { runPostprocess: false });
  } catch (error) {
    if (!expectedMessage.test(String(error.message))) throw error;
    cases.push(name);
    return;
  }
  throw new Error(`negative case unexpectedly passed: ${name}`);
}

await expectFailure('length error', /byte length/, async ({ capture, webBytes, temporary }) => {
  const truncated = webBytes.subarray(0, webBytes.length - 4);
  await writeFile(join(temporary, capture.webWasm.output.artifact.path), truncated);
  capture.webWasm.output.artifact.bytes -= 4;
  capture.webWasm.output.artifact.sha256 = sha256(truncated);
});
await expectFailure('extended length error', /byte length/, async ({ capture, webBytes, temporary }) => {
  const extended = Buffer.concat([webBytes, Buffer.alloc(4)]);
  await writeFile(join(temporary, capture.webWasm.output.artifact.path), extended);
  capture.webWasm.output.artifact.bytes = extended.length;
  capture.webWasm.output.artifact.sha256 = sha256(extended);
});
await expectFailure('SHA drift', /SHA-256/, async ({ capture }) => {
  capture.webWasm.output.artifact.sha256 = '0'.repeat(64);
});
await expectFailure('NaN', /non-finite value/, async ({ capture, webBytes, temporary }) => {
  webBytes.writeFloatLE(Number.NaN, 0);
  await writeFile(join(temporary, capture.webWasm.output.artifact.path), webBytes);
  capture.webWasm.output.artifact.sha256 = sha256(webBytes);
});
await expectFailure('Infinity', /non-finite value/, async ({ capture, nativeBytes, temporary }) => {
  nativeBytes.writeFloatLE(Number.POSITIVE_INFINITY, 0);
  await writeFile(join(temporary, capture.legacyNativeOrt.output.artifact.path), nativeBytes);
  capture.legacyNativeOrt.output.artifact.sha256 = sha256(nativeBytes);
});
await expectFailure('shape error', /dtype\/shape/, async ({ capture }) => {
  capture.webWasm.output.shape = [1, 8400, 84];
});
await expectFailure('dtype error', /dtype\/shape/, async ({ capture }) => {
  capture.webWasm.output.dtype = 'float16';
});
await expectFailure('endianness error', /endianness/, async ({ capture }) => {
  capture.webWasm.output.endianness = 'big-endian';
});
await expectFailure('product packaging overclaim', /packaging classification/, async ({ capture }) => {
  capture.webWasm.output.productPackaging = true;
});
await expectFailure('tampered comparison', /not recomputed/, async ({ capture }) => {
  capture.numericalComparison.maxAbsoluteDifference.index += 1;
});

await rm(temporary, { recursive: true, force: true });
console.log(JSON.stringify({ ok: true, negativeCases: cases }));
