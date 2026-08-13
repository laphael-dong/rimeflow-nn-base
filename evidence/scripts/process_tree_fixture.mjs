import { appendFileSync } from 'node:fs';
import { spawn } from 'node:child_process';

const [scenario, marker, depth = '0'] = process.argv.slice(2);
const append = () => appendFileSync(marker, `${process.pid}\n`);
const keepWriting = () => {
  append();
  setInterval(append, 15);
};
const spawnChild = (childScenario, childDepth) => spawn(process.execPath, [import.meta.filename, childScenario, marker, String(childDepth)], { stdio: 'ignore' });

if (scenario === 'success') {
  append();
} else if (scenario === 'tree-term') {
  if (Number(depth) < 2) spawnChild('tree-term', Number(depth) + 1);
  keepWriting();
} else if (scenario === 'tree-ignore-term') {
  process.on('SIGTERM', () => {});
  if (Number(depth) < 2) spawnChild('tree-ignore-term', Number(depth) + 1);
  keepWriting();
} else if (scenario === 'early-fail') {
  spawnChild('tree-ignore-term', 1);
  setTimeout(() => process.exit(7), 40);
} else if (scenario === 'zero-with-descendant') {
  spawnChild('tree-ignore-term', 1);
  setTimeout(() => process.exit(0), 40);
} else if (scenario === 'signal-with-descendant') {
  spawnChild('tree-ignore-term', 1);
  setTimeout(() => process.kill(process.pid, 'SIGHUP'), 40);
} else if (scenario === 'overflow-with-descendant') {
  spawnChild('tree-ignore-term', 1);
  process.stdout.write('x'.repeat(4096));
  setTimeout(() => process.exit(0), 40);
} else if (scenario === 'sentinel') {
  process.on('SIGTERM', () => process.exit(0));
  keepWriting();
} else {
  throw new Error(`unknown process-tree fixture scenario: ${scenario}`);
}
