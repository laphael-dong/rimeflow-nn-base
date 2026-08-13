import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const capturePath = resolve(root, 'evidence/performance/linux-x86_64-capture.json');
const outputPath = resolve(root, 'evidence/performance/linux-x86_64-baseline.json');
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const captureBytes = await readFile(capturePath);
const capture = JSON.parse(captureBytes);
const publication = JSON.parse(await readFile(resolve(root, 'evidence/publication/task1-publication.json'), 'utf8'));

if (capture.schemaVersion !== 2 || capture.numericalComparison?.passed !== true || capture.postprocessComparison?.comparison?.passed !== true) {
  throw new Error('refusing to generate a baseline from an invalid measurement capture');
}

const report = {
  ...capture,
  measurementIdentity: publication.measurementIdentity,
  operatorInputPublication: publication.operatorInputPublication,
  basePublicationState: publication.basePublicationState,
  source: {
    ...capture.source,
    measurementCapture: {
      path: 'evidence/performance/linux-x86_64-capture.json',
      sha256: sha256(captureBytes),
    },
    evidenceHarness: {
      ...capture.source.evidenceHarness,
      reportGeneratorSha256: sha256(await readFile(resolve(root, 'evidence/scripts/generate_performance_baseline.mjs'))),
    },
  },
};

await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(JSON.stringify({ output: 'evidence/performance/linux-x86_64-baseline.json', measurementCapture: report.source.measurementCapture }));
