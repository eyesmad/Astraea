// Finds the project's Python interpreter: the local venv if present, otherwise whatever is on PATH.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const root = path.join(path.dirname(fileURLToPath(import.meta.url)), '..');

export function pythonPath() {
  const win = process.platform === 'win32';
  const venv = path.join(root, 'venv', win ? 'Scripts/python.exe' : 'bin/python3');
  return process.env.PYTHON || (fs.existsSync(venv) ? venv : win ? 'python' : 'python3');
}
