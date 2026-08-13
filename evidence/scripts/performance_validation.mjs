import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';

export const OUTPUT_ELEMENTS = 705600;
export const OUTPUT_SHAPE = [1, 84, 8400];
export const OUTPUT_DTYPE = 'float32';
export const OUTPUT_ROLE = 'detections';
export const OUTPUT_NAME = 'output0';
export const ABSOLUTE_TOLERANCE = 1.0e-5;
export const RELATIVE_TOLERANCE = 1.0e-4;
export const RELATIVE_NEAR_ZERO_THRESHOLD = 1.0e-6;

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const sameJson = (left, right) => JSON.stringify(left) === JSON.stringify(right);

export async function readRawOutput(root, output) {
  if (output.logicalRole !== OUTPUT_ROLE || output.runtimeName !== OUTPUT_NAME) throw new Error('output logical role/name mismatch');
  if (output.dtype !== OUTPUT_DTYPE || output.endianness !== 'little-endian' || !sameJson(output.shape, OUTPUT_SHAPE)) throw new Error('output dtype/shape/endianness mismatch');
  if (output.classification !== 'raw-tensor-test-evidence' || output.productPackaging !== false) throw new Error('output packaging classification mismatch');
  if (output.elements !== OUTPUT_ELEMENTS) throw new Error('output element metadata mismatch');
  const path = resolve(root, output.artifact.path);
  const bytes = await readFile(path);
  if (bytes.length !== OUTPUT_ELEMENTS * 4 || output.artifact.bytes !== bytes.length) throw new Error('output byte length mismatch');
  if (output.artifact.sha256 !== sha256(bytes)) throw new Error('output artifact SHA-256 mismatch');
  const values = new Float32Array(OUTPUT_ELEMENTS);
  let finiteCount = 0;
  for (let index = 0; index < OUTPUT_ELEMENTS; index += 1) {
    const value = bytes.readFloatLE(index * 4);
    if (!Number.isFinite(value)) throw new Error(`output contains non-finite value at index ${index}`);
    values[index] = value;
    finiteCount += 1;
  }
  if (output.finiteCount !== finiteCount) throw new Error('output finite count mismatch');
  return values;
}

export function computeNumericalComparison(web, native) {
  if (web.length !== OUTPUT_ELEMENTS || native.length !== OUTPUT_ELEMENTS || web.length !== native.length) throw new Error('output element count mismatch');
  let toleranceMismatchCount = 0;
  let maxAbsolute = { difference: -1, index: -1 };
  let maxRelative = { difference: -1, index: -1 };
  let nearZeroCount = 0;
  let nearZeroMaxAbsolute = { difference: -1, index: -1 };
  for (let index = 0; index < OUTPUT_ELEMENTS; index += 1) {
    const webValue = web[index];
    const nativeValue = native[index];
    if (!Number.isFinite(webValue) || !Number.isFinite(nativeValue)) throw new Error(`non-finite comparison value at index ${index}`);
    const difference = Math.abs(nativeValue - webValue);
    const tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * Math.abs(webValue);
    if (difference > tolerance) toleranceMismatchCount += 1;
    if (difference > maxAbsolute.difference) maxAbsolute = { difference, index, webValue, nativeValue, tolerance };
    if (Math.abs(webValue) < RELATIVE_NEAR_ZERO_THRESHOLD) {
      nearZeroCount += 1;
      if (difference > nearZeroMaxAbsolute.difference) nearZeroMaxAbsolute = { difference, index, webValue, nativeValue, tolerance };
    } else {
      const relativeDifference = difference / Math.abs(webValue);
      if (relativeDifference > maxRelative.difference) maxRelative = { difference: relativeDifference, absoluteDifference: difference, index, webValue, nativeValue };
    }
  }
  const locate = (item) => ({
    ...item,
    attribute: Math.floor(item.index / 8400),
    anchor: item.index % 8400,
  });
  return {
    reference: 'webWasm',
    candidate: 'legacyNativeOrt',
    rule: 'abs(native-web) <= 1e-5 + 1e-4 * abs(web)',
    absoluteTolerance: ABSOLUTE_TOLERANCE,
    relativeTolerance: RELATIVE_TOLERANCE,
    maxAbsoluteDifference: locate(maxAbsolute),
    relativeDifference: {
      nearZeroThreshold: RELATIVE_NEAR_ZERO_THRESHOLD,
      maximumExcludingNearZero: maxRelative.index < 0 ? null : locate(maxRelative),
      nearZero: {
        count: nearZeroCount,
        maximumAbsoluteDifference: nearZeroMaxAbsolute.index < 0 ? null : locate(nearZeroMaxAbsolute),
      },
    },
    toleranceMismatchCount,
    passed: toleranceMismatchCount === 0,
  };
}

export function bboxIou(left, right) {
  const x1 = Math.max(left[0], right[0]);
  const y1 = Math.max(left[1], right[1]);
  const x2 = Math.min(left[2], right[2]);
  const y2 = Math.min(left[3], right[3]);
  const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const leftArea = Math.max(0, left[2] - left[0]) * Math.max(0, left[3] - left[1]);
  const rightArea = Math.max(0, right[2] - right[0]) * Math.max(0, right[3] - right[1]);
  const union = leftArea + rightArea - intersection;
  return union <= 0 ? 0 : intersection / union;
}

export function compareDetections(web, native, tolerances) {
  if (tolerances.classIdExact !== true || tolerances.confidenceAbsolute !== 1.0e-4 || tolerances.boxIouMinimum !== 0.999) throw new Error('postprocess tolerances do not match the frozen task 1 contract');
  if (web.length !== native.length) return { detectionCountEqual: false, webCount: web.length, nativeCount: native.length, pairs: [], passed: false };
  const pairs = web.map((reference, index) => {
    const candidate = native[index];
    const iou = bboxIou(reference.bbox, candidate.bbox);
    const confidenceDifference = Math.abs(reference.score - candidate.score);
    const classIdEqual = reference.classId === candidate.classId;
    const passed = classIdEqual && confidenceDifference <= tolerances.confidenceAbsolute && iou >= tolerances.boxIouMinimum;
    return { index, classIdEqual, webClassId: reference.classId, nativeClassId: candidate.classId, confidenceDifference, bboxIou: iou, passed };
  });
  return { detectionCountEqual: true, webCount: web.length, nativeCount: native.length, pairs, passed: pairs.every((item) => item.passed) };
}

export function runProductionPostprocess(operatorRoot, rawPath, width, height, outputPath, runnerPath = null, runnerEnv = process.env) {
  const executable = runnerPath ?? 'cargo';
  const args = runnerPath
    ? [rawPath, String(width), String(height), outputPath]
    : ['run', '--release', '--offline', '--manifest-path', resolve(operatorRoot, 'evidence/tooling/raw-golden/Cargo.toml'), '--', rawPath, String(width), String(height), outputPath];
  const result = spawnSync(executable, args, { cwd: operatorRoot, env: runnerEnv, encoding: 'utf8' });
  if (result.status !== 0) throw new Error(`production postprocess failed:\n${result.stdout}\n${result.stderr}`);
}

export async function generateProductionPostprocess(operatorRoot, artifactRoot, rawArtifact, width, height, outputPath) {
  await mkdir(dirname(outputPath), { recursive: true });
  runProductionPostprocess(operatorRoot, resolve(artifactRoot, rawArtifact.path), width, height, outputPath);
  return JSON.parse(await readFile(outputPath, 'utf8'));
}

export async function validatePerformanceCapture(root, operatorRoot, capture, { runPostprocess = true, productionRunner = null, productionRunnerEnv = process.env, scratchRoot = null } = {}) {
  if (capture.schemaVersion !== 2) throw new Error('performance capture schema mismatch');
  const providers = capture.comparability?.providerIdentity;
  if (!capture.comparability?.sameDevice || capture.comparability?.crossBackendProviderEqual !== false) throw new Error('mixed device/cross-backend provider semantics');
  if (providers?.web?.provider !== 'wasm' || providers.web.stableAcrossRounds !== true || providers.web.roundProviders.some((provider) => provider !== 'wasm')) throw new Error('mixed Web execution provider rounds');
  if (providers?.native?.provider !== 'Cpu' || providers.native.stableAcrossRounds !== true || providers.native.roundProviders.some((provider) => provider !== 'Cpu')) throw new Error('mixed Native execution provider rounds');
  const web = await readRawOutput(root, capture.webWasm.output);
  const native = await readRawOutput(root, capture.legacyNativeOrt.output);
  const numerical = computeNumericalComparison(web, native);
  if (!sameJson(numerical, capture.numericalComparison)) throw new Error('reported numerical comparison was not recomputed from raw artifacts');
  if (!numerical.passed) throw new Error('raw numerical comparison failed');
  if (runPostprocess) {
    const scratch = scratchRoot ?? resolve(root, '.evidence/task1-linux-baseline/validator-postprocess');
    await mkdir(scratch, { recursive: true });
    const webPath = resolve(scratch, 'web.json');
    const nativePath = resolve(scratch, 'native.json');
    const width = capture.source.fixtureGeometry.width;
    const height = capture.source.fixtureGeometry.height;
    runProductionPostprocess(operatorRoot, resolve(root, capture.webWasm.output.artifact.path), width, height, webPath, productionRunner, productionRunnerEnv);
    runProductionPostprocess(operatorRoot, resolve(root, capture.legacyNativeOrt.output.artifact.path), width, height, nativePath, productionRunner, productionRunnerEnv);
    const webDetections = JSON.parse(await readFile(webPath, 'utf8'));
    const nativeDetections = JSON.parse(await readFile(nativePath, 'utf8'));
    if (!sameJson(webDetections, capture.postprocessComparison.webDetections) || !sameJson(nativeDetections, capture.postprocessComparison.nativeDetections)) throw new Error('stored production postprocess output drift');
    const comparison = compareDetections(webDetections, nativeDetections, capture.postprocessComparison.tolerances);
    if (!sameJson(comparison, capture.postprocessComparison.comparison) || !comparison.passed) throw new Error('reported production postprocess comparison mismatch');
  }
  return { numericalComparison: numerical, postprocessComparison: capture.postprocessComparison.comparison };
}

export async function writeF32Artifact(path, values) {
  const bytes = Buffer.alloc(values.length * 4);
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) throw new Error(`refusing to write non-finite output at index ${index}`);
    bytes.writeFloatLE(value, index * 4);
  }
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, bytes);
  return { bytes: bytes.length, sha256: sha256(bytes) };
}
