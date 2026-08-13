import { createHash } from 'node:crypto';
import { chmod, lstat, mkdir, readFile, readdir, realpath, rename, rm, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';

import { OFFICIAL_LINUX_X64_TOOLS, runTrusted, runTrustedProcessGroup } from './official_live_trust.mjs';

const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const gitBlobId = (bytes) => createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
const receiptName = 'operator-input-receipt.json';
const safePath = (path) => path.length > 0 && !path.startsWith('/') && !path.includes('\\') && !path.split('/').includes('..');
const sameKeys = (value, keys) => value !== null && typeof value === 'object' && !Array.isArray(value) && JSON.stringify(Object.keys(value).sort()) === JSON.stringify([...keys].sort());

const validateSuccessfulFetchProcessGroup = (processGroup) => {
  const keys = ['program', 'args', 'leaderPid', 'pgid', 'timedOut', 'exitCode', 'exitSignal', 'spawnError', 'outputOverflow', 'sigterm', 'sigkill', 'closed', 'membersBeforeCleanup', 'membersAfterTerm', 'membersAfterKill', 'remainingMembers', 'cleanupComplete'];
  if (!sameKeys(processGroup, keys)) throw new Error('operator export fetch process-group receipt fields mismatch');
  if (processGroup.program !== OFFICIAL_LINUX_X64_TOOLS.git.path || !Number.isSafeInteger(processGroup.leaderPid) || processGroup.leaderPid <= 1 || processGroup.pgid !== processGroup.leaderPid) throw new Error('operator export fetch process-group leader identity mismatch');
  if (processGroup.timedOut !== false || processGroup.exitCode !== 0 || processGroup.exitSignal !== null || processGroup.spawnError !== null || processGroup.outputOverflow !== false || processGroup.closed !== true || processGroup.cleanupComplete !== true) throw new Error('operator export fetch process-group success state mismatch');
  if (processGroup.sigterm.attempted !== false || processGroup.sigkill.attempted !== false || ![processGroup.membersBeforeCleanup, processGroup.membersAfterTerm, processGroup.membersAfterKill, processGroup.remainingMembers].every((members) => Array.isArray(members) && members.length === 0)) throw new Error('operator export fetch process-group cleanup state mismatch');
};

const listTree = (repository, git, gitEnv) => {
  const bytes = runTrusted(git, ['ls-tree', '-rz', '--full-tree', 'FETCH_HEAD'], { cwd: repository, env: gitEnv, encoding: 'buffer' });
  return bytes.toString().split('\0').filter(Boolean).map((record) => {
    const tab = record.indexOf('\t');
    if (tab < 0) throw new Error('operator tree record missing path separator');
    const [mode, type, blob] = record.slice(0, tab).split(' ');
    const path = record.slice(tab + 1);
    if (!safePath(path) || type !== 'blob' || !['100644', '100755'].includes(mode)) throw new Error(`operator tree contains unsupported entry: ${path}`);
    return { mode, type, blob, path };
  });
};

const listSourceFiles = async (source) => {
  const files = [];
  const visit = async (directory) => {
    for (const name of await readdir(directory)) {
      const path = resolve(directory, name);
      const metadata = await lstat(path);
      if (metadata.isSymbolicLink()) throw new Error(`operator source contains symlink: ${path}`);
      if (metadata.isDirectory()) await visit(path);
      else if (metadata.isFile()) files.push(relative(source, path));
      else throw new Error(`operator source contains unsupported file type: ${path}`);
    }
  };
  await visit(source);
  return files.sort();
};

export async function verifyOperatorExport(bundleRoot, publication, { git = OFFICIAL_LINUX_X64_TOOLS.git.path, gitEnv } = {}) {
  const bundle = await realpath(bundleRoot);
  const repository = resolve(bundle, 'operator.git');
  const source = resolve(bundle, 'source');
  const receipt = JSON.parse(await readFile(resolve(bundle, receiptName), 'utf8'));
  const input = publication.operatorInputPublication;
  if (!sameKeys(receipt, ['schemaVersion', 'repository', 'ref', 'commit', 'tree', 'source', 'exportedFileCount', 'fetchProcessGroup']) || receipt.schemaVersion !== 1 || receipt.source !== 'fresh-bare-fetch-and-complete-blob-export') throw new Error('operator export receipt fields mismatch');
  if (receipt.repository !== input.repository || receipt.ref !== input.ref || receipt.commit !== input.commit || receipt.tree !== input.tree) throw new Error('operator export receipt identity mismatch');
  validateSuccessfulFetchProcessGroup(receipt.fetchProcessGroup);
  if (JSON.stringify(receipt.fetchProcessGroup.args) !== JSON.stringify(['fetch', '--no-tags', '--depth=1', input.repository, input.ref])) throw new Error('operator export fetch process-group arguments mismatch');
  const fetchHead = runTrusted(git, ['rev-parse', 'FETCH_HEAD'], { cwd: repository, env: gitEnv }).trim();
  const tree = runTrusted(git, ['rev-parse', 'FETCH_HEAD^{tree}'], { cwd: repository, env: gitEnv }).trim();
  if (fetchHead !== input.commit || tree !== input.tree) throw new Error('operator export repository commit/tree mismatch');
  const treeEntries = listTree(repository, git, gitEnv);
  const sourcePaths = await listSourceFiles(source);
  if (JSON.stringify(sourcePaths) !== JSON.stringify(treeEntries.map((entry) => entry.path).sort())) throw new Error('operator export complete source path set mismatch');
  for (const entry of treeEntries) {
    const path = resolve(source, entry.path);
    const metadata = await lstat(path);
    if (!metadata.isFile() || metadata.isSymbolicLink()) throw new Error(`operator source entry is not a regular file: ${entry.path}`);
    const bytes = await readFile(path);
    if (gitBlobId(bytes) !== entry.blob) throw new Error(`operator source Git blob identity mismatch: ${entry.path}`);
    if ((entry.mode === '100755') !== ((metadata.mode & 0o111) !== 0)) throw new Error(`operator source executable mode mismatch: ${entry.path}`);
  }
  const verified = [];
  for (const object of [...input.objects, ...input.productionSources]) {
    if (!safePath(object.path)) throw new Error(`operator export unsafe path: ${object.id}`);
    const entry = runTrusted(git, ['ls-tree', 'FETCH_HEAD', '--', object.path], { cwd: repository, env: gitEnv }).trim().split(/\s+/);
    if (entry[0] !== object.mode || entry[1] !== object.type || entry[2] !== object.blob) throw new Error(`operator export Git tuple mismatch: ${object.id}`);
    const path = resolve(source, object.path);
    const canonical = await realpath(path);
    if (!canonical.startsWith(`${source}/`)) throw new Error(`operator export path escaped source: ${object.id}`);
    const metadata = await lstat(path);
    if (!metadata.isFile() || metadata.isSymbolicLink()) throw new Error(`operator export object is not a regular file: ${object.id}`);
    const bytes = await readFile(path);
    const digest = sha256(bytes);
    if (bytes.length !== object.bytes || digest !== object.sha256) throw new Error(`operator export bytes/SHA-256 mismatch: ${object.id}`);
    verified.push({ id: object.id, path: object.path, mode: object.mode, type: object.type, blob: object.blob, bytes: bytes.length, sha256: digest });
  }
  if (receipt.exportedFileCount !== treeEntries.length) throw new Error('operator export receipt file count mismatch');
  return { bundle, repository, source, receipt, fetchHead, tree, exportedFileCount: treeEntries.length, verified };
}

async function makeReadOnly(root) {
  const entries = [];
  const visit = async (path) => {
    const metadata = await lstat(path);
    if (metadata.isSymbolicLink()) throw new Error(`read-only source contains symlink: ${path}`);
    if (metadata.isDirectory()) {
      for (const name of await readdir(path)) await visit(join(path, name));
      await chmod(path, 0o555);
      entries.push({ path, type: 'directory', mode: '555' });
    } else if (metadata.isFile()) {
      const executable = (metadata.mode & 0o111) !== 0;
      await chmod(path, executable ? 0o555 : 0o444);
      entries.push({ path, type: 'file', mode: executable ? '555' : '444' });
    } else throw new Error(`read-only source contains unsupported file type: ${path}`);
  };
  await visit(root);
  for (const entry of entries) if (((await lstat(entry.path)).mode & 0o222) !== 0) throw new Error(`source remained writable: ${entry.path}`);
  return { verifiedEntryCount: entries.length, allEntriesNonWritable: true };
}

export async function createOperatorExport(bundleRoot, publication, { git = OFFICIAL_LINUX_X64_TOOLS.git.path, gitEnv, readOnly = false } = {}) {
  const output = resolve(bundleRoot);
  try { await lstat(output); throw new Error(`operator export target already exists: ${output}`); } catch (error) { if (error.code !== 'ENOENT') throw error; }
  const temporary = `${output}.partial-${process.pid}`;
  await mkdir(dirname(output), { recursive: true });
  await mkdir(temporary, { mode: 0o700 });
  const repository = resolve(temporary, 'operator.git');
  const source = resolve(temporary, 'source');
  try {
    await mkdir(source);
    runTrusted(git, ['init', '--bare', repository], { env: gitEnv });
    const input = publication.operatorInputPublication;
    const fetchProcess = await runTrustedProcessGroup(git, ['fetch', '--no-tags', '--depth=1', input.repository, input.ref], { cwd: repository, env: gitEnv, inherit: true, timeoutMs: 300000, termGraceMs: 2000, killGraceMs: 2000, closeGraceMs: 2000 });
    const fetchHead = runTrusted(git, ['rev-parse', 'FETCH_HEAD'], { cwd: repository, env: gitEnv }).trim();
    const tree = runTrusted(git, ['rev-parse', 'FETCH_HEAD^{tree}'], { cwd: repository, env: gitEnv }).trim();
    if (fetchHead !== input.commit || tree !== input.tree) throw new Error(`operator fetch commit/tree mismatch: ${fetchHead}/${tree}`);
    const treeEntries = listTree(repository, git, gitEnv);
    for (const entry of treeEntries) {
      const bytes = runTrusted(git, ['cat-file', 'blob', entry.blob], { cwd: repository, env: gitEnv, encoding: 'buffer' });
      if (gitBlobId(bytes) !== entry.blob) throw new Error(`operator fetched Git blob identity mismatch: ${entry.path}`);
      const destination = resolve(source, entry.path);
      await mkdir(dirname(destination), { recursive: true });
      await writeFile(destination, bytes, { mode: entry.mode === '100755' ? 0o700 : 0o600 });
    }
    const receipt = { schemaVersion: 1, repository: input.repository, ref: input.ref, commit: fetchHead, tree, source: 'fresh-bare-fetch-and-complete-blob-export', exportedFileCount: treeEntries.length, fetchProcessGroup: fetchProcess.processGroup };
    await writeFile(resolve(temporary, receiptName), `${JSON.stringify(receipt, null, 2)}\n`, { mode: 0o600 });
    const prepared = await verifyOperatorExport(temporary, publication, { git, gitEnv });
    const sourcePermissions = readOnly ? await makeReadOnly(prepared.source) : null;
    await rename(temporary, output);
    const verified = await verifyOperatorExport(output, publication, { git, gitEnv });
    return { ...verified, sourcePermissions };
  } catch (error) {
    const cleanupErrors = [];
    for (const path of [temporary, output]) {
      try { await rm(path, { recursive: true, force: true }); } catch (cleanupError) { cleanupErrors.push(`${path}: ${cleanupError.code ?? cleanupError.name}: ${cleanupError.message}`); }
    }
    if (cleanupErrors.length > 0) throw new Error(`${error.message}; operator export cleanup failed: ${JSON.stringify(cleanupErrors)}`, { cause: error });
    throw error;
  }
}

export async function restoreOwnerWrite(root) {
  const visit = async (path) => {
    const metadata = await lstat(path);
    if (metadata.isDirectory()) {
      await chmod(path, (metadata.mode & 0o777) | 0o700);
      for (const name of await readdir(path)) await visit(join(path, name));
    } else if (metadata.isFile()) await chmod(path, (metadata.mode & 0o777) | 0o600);
  };
  try { await visit(root); } catch (error) { if (error.code !== 'ENOENT') throw error; }
}
