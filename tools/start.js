// Starts both halves of Astraea: the Node web server and the Python analysis server it proxies /ws/analyze to.
//
// This exists because `node server.js & python ...` only backgrounds the first command on POSIX shells; on
// Windows npm runs scripts through cmd.exe, where `&` is sequential, so the Python half never started.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');
const win = process.platform === 'win32';
const venv = path.join(root, 'venv', win ? 'Scripts/python.exe' : 'bin/python3');
const python = process.env.PYTHON || (fs.existsSync(venv) ? venv : win ? 'python' : 'python3');
const watch = process.argv.includes('--watch');

const children = [];
let shuttingDown = false;

function stopAll(code) {
  if (shuttingDown) return;
  shuttingDown = true;
  for (const c of children) c.kill();
  process.exit(code);
}

function run(name, cmd, args) {
  const child = spawn(cmd, args, { cwd: root, stdio: 'inherit' });
  child.on('error', (err) => {
    console.error(`[${name}] could not start: ${err.message}`);
    if (name === 'python') console.error('[python] install its dependencies: python -m venv venv && venv/Scripts/pip install -r server/requirements.txt');
    stopAll(1);
  });
  child.on('exit', (code) => {
    if (!shuttingDown) console.error(`[${name}] exited (${code}); stopping the other half.`);
    stopAll(code ?? 1);
  });
  children.push(child);
}

run('node', process.execPath, watch ? ['--watch', 'server.js'] : ['server.js']);
run('python', python, ['server/analysis_server.py']);

for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => stopAll(0));
