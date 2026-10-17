// Replays a recording through AnalysisSession and reports what each signal did.
// Evaluation is driven by the recorded clock, not wall time, so a replay is deterministic and runs as fast
// as the CPU allows: the same recording always produces the same timeline.
import { sessionFor } from './analysis.js';
import { decodeAudioChunk, readRecording } from './recording.js';

export const TICK_MS = 1000; // matches the live socket's evaluation cadence

export async function replay(file, { tickMs = TICK_MS } = {}) {
  let header = null;
  let session = null;
  let nextTick = null;
  const ticks = [];
  const signals = new Map();

  const runTick = (t) => {
    const results = session.evaluate();
    ticks.push({ t, fired: results.filter((r) => r.fired).map((r) => r.id) });
    for (const r of results) {
      if (!signals.has(r.id)) signals.set(r.id, { id: r.id, evaluated: 0, fired: 0, firstFiredT: null, maxSeverity: 0, lastDetail: '' });
      const s = signals.get(r.id);
      s.evaluated++;
      s.lastDetail = r.detail || '';
      if (r.fired) {
        s.fired++;
        s.maxSeverity = Math.max(s.maxSeverity, r.severity || 0);
        if (s.firstFiredT == null) s.firstFiredT = t;
      }
    }
  };

  for await (const rec of readRecording(file)) {
    if (rec.type === 'header') {
      header = rec;
      session = sessionFor(rec.sampleRate);
      continue;
    }
    if (rec.type === 'audio') {
      const chunk = decodeAudioChunk(Buffer.from(rec.b64, 'base64'));
      if (chunk) session.pushAudio(chunk.samples, chunk.endT);
    } else if (rec.type === 'face') {
      session.pushFace(rec.t, rec.p);
    } else {
      continue;
    }
    // Start the tick clock at the first real measurement: a session has no clock until enough samples have
    // arrived to fill one analysis frame, and ticking from zero would invent evaluations for that dead time.
    if (nextTick == null) {
      if (session.now <= 0) continue;
      nextTick = session.now + tickMs;
    }
    while (session.now >= nextTick) {
      await session.drainModel();
      runTick(nextTick);
      nextTick += tickMs;
    }
  }
  if (!session) throw new Error(`${file}: no header`);

  return {
    file,
    header,
    seconds: ticks.length ? (ticks[ticks.length - 1].t - ticks[0].t + tickMs) / 1000 : 0,
    frames: session.frames,
    ticks,
    signals: [...signals.values()].sort((a, b) => b.fired - a.fired),
    probes: session.probes(),
  };
}

// Aggregates several replays. A recording labelled `genuine` is a real person, so any signal that fires on it
// is a false positive; `fake` recordings are the opposite. Everything else is left unlabelled and only counted.
export function summarize(runs) {
  const rows = new Map();
  for (const run of runs) {
    const label = run.header.label === 'genuine' || run.header.label === 'fake' ? run.header.label : 'unlabelled';
    for (const s of run.signals) {
      if (!rows.has(s.id)) rows.set(s.id, { id: s.id, genuine: { ticks: 0, fired: 0, files: 0 }, fake: { ticks: 0, fired: 0, files: 0 }, unlabelled: { ticks: 0, fired: 0, files: 0 } });
      const cell = rows.get(s.id)[label];
      cell.ticks += s.evaluated;
      cell.fired += s.fired;
      if (s.fired > 0) cell.files++;
    }
  }
  return [...rows.values()].sort((a, b) => b.genuine.fired / (b.genuine.ticks || 1) - a.genuine.fired / (a.genuine.ticks || 1));
}
