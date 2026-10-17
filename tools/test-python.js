// Runs the Python side of the suite, so `npm test` covers both halves of the app.
import { spawnSync } from 'node:child_process';
import { pythonPath, root } from './python.js';

const r = spawnSync(pythonPath(), ['-m', 'unittest', 'discover', '-s', 'tests'], { cwd: root, stdio: 'inherit' });
if (r.error) {
  console.error(`\nCould not run the Python tests: ${r.error.message}`);
  console.error('Set up its environment first: python -m venv venv && venv/Scripts/pip install -r server/requirements.txt');
}
process.exit(r.status ?? 1);
