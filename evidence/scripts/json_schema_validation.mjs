import { createHash } from 'node:crypto';

const fail = (path, message) => { throw new Error(`JSON schema ${path}: ${message}`); };
const typeMatches = (value, type) => type === 'object' ? value !== null && typeof value === 'object' && !Array.isArray(value) : type === 'array' ? Array.isArray(value) : type === 'number' ? typeof value === 'number' && Number.isFinite(value) : type === 'integer' ? Number.isInteger(value) : type === 'string' ? typeof value === 'string' : type === 'boolean' ? typeof value === 'boolean' : type === 'null' ? value === null : false;
export const PLATFORM_SCHEMA_SHA256 = '3fd8927707a794618f710c50469c83ab2dac8ad82793829ccf0d4552653f92ad';
const SUPPORTED_KEYWORDS = new Set(['$schema', '$id', '$ref', 'type', 'additionalProperties', 'required', 'properties', 'const', 'enum', 'minimum', 'minLength', 'pattern', 'format', 'minItems', 'items', 'definitions']);

const validCalendarDate = (value) => {
  const match = /^(\d{4})-(\d{2})-(\d{2})$/.exec(value);
  if (!match) return false;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const date = new Date(Date.UTC(year, month - 1, day));
  return date.getUTCFullYear() === year && date.getUTCMonth() === month - 1 && date.getUTCDate() === day;
};

export function validateFrozenPlatformSchema(bytes, schema) {
  const audit = (value, path) => {
    if (Array.isArray(value)) return value.forEach((item, index) => audit(item, `${path}[${index}]`));
    if (value === null || typeof value !== 'object') return;
    for (const [key, child] of Object.entries(value)) {
      if (!SUPPORTED_KEYWORDS.has(key) && path !== '$.properties' && !path.endsWith('.properties') && path !== '$.definitions' && !path.endsWith('.definitions')) fail(`${path}.${key}`, 'unsupported Schema keyword');
      audit(child, `${path}.${key}`);
    }
  };
  audit(schema, '$');
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (digest !== PLATFORM_SCHEMA_SHA256) fail('$', `frozen platform Schema SHA-256 mismatch: ${digest}`);
  return { bytes: bytes.length, sha256: digest, draft: schema.$schema, id: schema.$id };
}

export function validateJsonSchema(schema, value) {
  const resolveRef = (ref) => {
    if (!ref.startsWith('#/')) fail('$ref', `unsupported reference ${ref}`);
    return ref.slice(2).split('/').reduce((current, part) => current?.[part.replaceAll('~1', '/').replaceAll('~0', '~')], schema);
  };
  const visit = (rule, current, path) => {
    if (rule.$ref) return visit(resolveRef(rule.$ref), current, path);
    if ('const' in rule && JSON.stringify(current) !== JSON.stringify(rule.const)) fail(path, 'const mismatch');
    if (rule.enum && !rule.enum.some((item) => JSON.stringify(item) === JSON.stringify(current))) fail(path, 'enum mismatch');
    if (rule.type && !typeMatches(current, rule.type)) fail(path, `expected ${rule.type}`);
    if (typeof current === 'number' && rule.minimum !== undefined && current < rule.minimum) fail(path, `minimum ${rule.minimum}`);
    if (typeof current === 'string') {
      if (rule.minLength !== undefined && current.length < rule.minLength) fail(path, `minLength ${rule.minLength}`);
      if (rule.pattern && !new RegExp(rule.pattern).test(current)) fail(path, 'pattern mismatch');
      if (rule.format && rule.format !== 'date') fail(path, `unsupported format ${rule.format}`);
      if (rule.format === 'date' && !validCalendarDate(current)) fail(path, 'calendar date format');
    }
    if (Array.isArray(current)) {
      if (rule.minItems !== undefined && current.length < rule.minItems) fail(path, `minItems ${rule.minItems}`);
      if (rule.items) current.forEach((item, index) => visit(rule.items, item, `${path}[${index}]`));
    } else if (current !== null && typeof current === 'object') {
      for (const key of rule.required ?? []) if (!(key in current)) fail(`${path}.${key}`, 'required property missing');
      if (rule.additionalProperties === false) for (const key of Object.keys(current)) if (!(key in (rule.properties ?? {}))) fail(`${path}.${key}`, 'additional property');
      for (const [key, childRule] of Object.entries(rule.properties ?? {})) if (key in current) visit(childRule, current[key], `${path}.${key}`);
    }
  };
  visit(schema, value, '$');
  return true;
}
