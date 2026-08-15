import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const root = process.cwd();
const schema = JSON.parse(readFileSync(join(root, 'schemas/model-manifest.schema.json'), 'utf8'));
const fixtures = [
  ['valid-yolov8n.json', true],
  ['invalid-structure.json', false],
  ['unknown-schema-version.json', false],
  ['artifact-integrity-mismatch.json', false],
  ['invalid-quantization.json', false],
];

function fail(path, message) {
  throw new Error(`model manifest schema ${path}: ${message}`);
}

function resolveRef(ref) {
  if (!ref.startsWith('#/')) fail('$ref', `unsupported reference ${ref}`);
  return ref.slice(2).split('/').reduce(
    (current, part) => current?.[part.replaceAll('~1', '/').replaceAll('~0', '~')],
    schema,
  );
}

function typeMatches(value, type) {
  if (type === 'object') return value !== null && typeof value === 'object' && !Array.isArray(value);
  if (type === 'array') return Array.isArray(value);
  if (type === 'number') return typeof value === 'number' && Number.isFinite(value);
  if (type === 'integer') return Number.isInteger(value);
  if (type === 'string') return typeof value === 'string';
  if (type === 'boolean') return typeof value === 'boolean';
  if (type === 'null') return value === null;
  return false;
}

function visit(rule, value, path) {
  if (rule.$ref) return visit(resolveRef(rule.$ref), value, path);
  if (rule.anyOf) {
    const accepted = rule.anyOf.filter((candidate) => matches(candidate, value, path)).length;
    if (accepted === 0) fail(path, 'no anyOf branch matched');
  }
  for (const nested of rule.allOf ?? []) visit(nested, value, path);
  if (rule.if && matches(rule.if, value, path) && rule.then) visit(rule.then, value, path);
  if ('const' in rule && JSON.stringify(value) !== JSON.stringify(rule.const)) {
    fail(path, 'const mismatch');
  }
  if (rule.enum && !rule.enum.some((candidate) => JSON.stringify(candidate) === JSON.stringify(value))) {
    fail(path, 'enum mismatch');
  }
  if (rule.type && !typeMatches(value, rule.type)) fail(path, `expected ${rule.type}`);
  if (typeof value === 'number') {
    if (rule.minimum !== undefined && value < rule.minimum) fail(path, `minimum ${rule.minimum}`);
    if (rule.exclusiveMinimum !== undefined && value <= rule.exclusiveMinimum) {
      fail(path, `exclusiveMinimum ${rule.exclusiveMinimum}`);
    }
  }
  if (typeof value === 'string') {
    if (rule.minLength !== undefined && value.length < rule.minLength) {
      fail(path, `minLength ${rule.minLength}`);
    }
    if (rule.pattern && !new RegExp(rule.pattern, 'u').test(value)) fail(path, 'pattern mismatch');
  }
  if (Array.isArray(value)) {
    if (rule.minItems !== undefined && value.length < rule.minItems) {
      fail(path, `minItems ${rule.minItems}`);
    }
    if (rule.items) value.forEach((item, index) => visit(rule.items, item, `${path}[${index}]`));
    return;
  }
  if (value !== null && typeof value === 'object') {
    for (const key of rule.required ?? []) {
      if (!(key in value)) fail(`${path}.${key}`, 'required property missing');
    }
    if (rule.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!(key in (rule.properties ?? {}))) fail(`${path}.${key}`, 'additional property');
      }
    }
    for (const [key, childRule] of Object.entries(rule.properties ?? {})) {
      if (key in value) visit(childRule, value[key], `${path}.${key}`);
    }
  }
}

function matches(rule, value, path) {
  try {
    visit(rule, value, path);
    return true;
  } catch {
    return false;
  }
}

for (const [name, expected] of fixtures) {
  const value = JSON.parse(
    readFileSync(join(root, 'tests/fixtures/manifest', name), 'utf8'),
  );
  const accepted = matches(schema, value, '$');
  if (accepted !== expected) {
    throw new Error(`model manifest fixture ${name}: expected schema accepted=${expected}, got ${accepted}`);
  }
}

console.log(`Model manifest Schema fixture conclusions valid: ${fixtures.length} fixtures.`);
