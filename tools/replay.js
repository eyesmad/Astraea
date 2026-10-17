#!/usr/bin/env node
// Replays recorded captures through the analysis pipeline.
//
//   node tools/replay.js recordings/genuine-2026-10-17.ndjson      one capture, with a timeline
//   node tools/replay.js recordings/*.ndjson --summary             how often each signal fires, by label
//   node tools/replay.js recordings/*.ndjson --json                machine-readable output
//
// A capture labelled `genuine` is a real person, so every tick a signal fires on it is a false positive.
// That number is what the thresholds should be tuned against.
import fs from 'node:fs';
import path from 'node:path';
import { replay, summarize } from '../server/replay.js';

const args = process.argv.slice(2);
const flags = new Set(args.filter((a) => a.startsWith('--')));
let files = args.filter((a) => !a.startsWith('--'));

if (!files.length) {
  const dir = 'recordings';
  files = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => f.endsWith('.ndjson')).map((f) => path.join(dir, f)) : [];
}
if (!files.length) {
  console.error('No recordings. Capture one first: open the app with ?record=genuine and start a capture.');
  process.exit(1);
}

const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(1)}%` : '-');
const runs = [];
for (const f of files) runs.push(await replay(f));

if (flags.has('--json')) {
  console.log(JSON.stringify(flags.has('--summary') ? summarize(runs) : runs.map(({ ticks, ...r }) => r), null, 1));
} else if (flags.has('--summary') || runs.length > 1) {
  const totals = { genuine: 0, fake: 0, unlabelled: 0 };
  for (const r of runs) {
    const label = r.header.label === 'genuine' || r.header.label === 'fake' ? r.header.label : 'unlabelled';
    totals[label] += r.seconds;
  }
  console.log(`${runs.length} recordings: ${Object.entries(totals).filter(([, s]) => s).map(([k, s]) => `${s.toFixed(0)} s ${k}`).join(', ')}\n`);
  console.log('signal'.padEnd(22) + 'fires on genuine'.padEnd(20) + 'fires on fake');
  console.log('-'.repeat(60));
  for (const row of summarize(runs)) {
    const g = `${pct(row.genuine.fired, row.genuine.ticks)} (${row.genuine.files} files)`;
    const k = `${pct(row.fake.fired, row.fake.ticks)} (${row.fake.files} files)`;
    console.log(row.id.padEnd(22) + g.padEnd(20) + k);
  }
  console.log('\nPercentages are the share of one-second evaluations where the signal fired.');
  console.log('On genuine captures that is the false-positive rate; aim well under 1%.');
} else {
  const r = runs[0];
  console.log(`${r.file}\n  label ${r.header.label || '(none)'}  ${r.seconds.toFixed(0)} s  ${r.frames} face frames  ${r.header.sampleRate} Hz`);
  if (r.header.note) console.log(`  note: ${r.header.note}`);

  console.log('\nTimeline (signals changing state):');
  let prev = new Set();
  for (const t of r.ticks) {
    const now = new Set(t.fired);
    const on = [...now].filter((x) => !prev.has(x));
    const off = [...prev].filter((x) => !now.has(x));
    if (on.length || off.length) {
      const s = `${((t.t - r.ticks[0].t) / 1000).toFixed(0)}s`.padStart(6);
      console.log(`${s}  ${[...on.map((x) => `+${x}`), ...off.map((x) => `-${x}`)].join(' ')}`);
    }
    prev = now;
  }
  if (!r.ticks.some((t) => t.fired.length)) console.log('  (nothing fired)');

  console.log('\nPer signal:');
  for (const s of r.signals) {
    console.log(`  ${s.id.padEnd(20)} fired ${String(s.fired).padStart(4)}/${String(s.evaluated).padEnd(5)} ${pct(s.fired, s.evaluated).padStart(7)}  ${s.lastDetail}`);
  }

  console.log('\nFinal readings:');
  for (const [layer, rows] of Object.entries(r.probes)) {
    for (const p of rows) console.log(`  ${layer.padEnd(12)} ${p.label.padEnd(24)} ${p.text}`);
  }
}
