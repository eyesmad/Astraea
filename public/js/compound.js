// Compound marker: two or more independent LIVE signals inside a rolling window.
// Standing signals (metadata, content) do not need to fall inside the window; they corroborate and raise strength.
import { SIGNALS, SYNERGY } from './signals.js';

export class CompoundDetector {
  constructor({ windowMs = 10000, minSignals = 2, cooldownMs = 6000 } = {}) {
    this.windowMs = windowMs;
    this.minSignals = minSignals;
    this.cooldownMs = cooldownMs;
    this.live = new Map(); // id -> { t, severity }
    this.standing = new Map(); // id -> { severity }
    this.lastEmit = { t: -Infinity, key: '' };
  }

  reset() {
    this.live.clear();
    this.standing.clear();
    this.lastEmit = { t: -Infinity, key: '' };
  }

  // Report a signal at time t (ms). Pass severity 0 (or fired false) to clear a standing signal.
  report(id, { fired, severity }, t) {
    const def = SIGNALS[id];
    if (!def) return;
    if (def.standing) {
      if (fired) this.standing.set(id, { severity });
      else this.standing.delete(id);
      return;
    }
    if (fired) this.live.set(id, { t, severity });
    else this.live.delete(id);
  }

  // Returns a marker object when a new compound event should be announced, else null.
  evaluate(now) {
    for (const [id, s] of this.live) if (now - s.t > this.windowMs) this.live.delete(id);
    const ids = [...this.live.keys()].sort();
    if (ids.length < this.minSignals) return null;

    const layers = new Set(ids.map((id) => SIGNALS[id].layer));
    const standingIds = [...this.standing.keys()].sort();
    const synergy = SYNERGY.filter(([a, b]) => this.live.has(a) && this.live.has(b));

    // Noisy-or over weighted severities, so independent evidence compounds.
    let miss = 1;
    for (const id of ids) miss *= 1 - SIGNALS[id].weight * this.live.get(id).severity;
    for (const id of standingIds) miss *= 1 - 0.5 * SIGNALS[id].weight * this.standing.get(id).severity;
    for (let i = 0; i < synergy.length; i++) miss *= 0.85;
    const strength = Math.min(1, 1 - miss);

    const key = ids.join('+') + '|' + standingIds.join('+');
    const isNew = key !== this.lastEmit.key;
    if (!isNew && now - this.lastEmit.t < this.cooldownMs) return null;
    if (isNew && now - this.lastEmit.t < 1500) return null;
    this.lastEmit = { t: now, key };

    return {
      t: now,
      signals: ids,
      layers: [...layers],
      standing: standingIds,
      synergy: synergy.map(([a, b]) => `${a} + ${b}`),
      strength,
    };
  }
}

// Trust is 1.0 for a clean call and falls as independent evidence accumulates.
export function computeTrust(active, markerStrength = 0) {
  let trust = 1;
  for (const { id, severity } of active) {
    const def = SIGNALS[id];
    if (def) trust *= 1 - def.weight * severity;
  }
  trust *= 1 - 0.25 * markerStrength;
  return Math.max(0, Math.min(1, trust));
}

// Evidence seen so far in the call, not just what is firing right now.
//
// A caller who looked synthetic for twenty seconds should not read as trustworthy the moment the signal
// stops: whoever was on the line is still the person you were talking to. Each signal keeps a score that
// accumulates while it fires (repeated evidence compounds rather than only counting the worst moment) and
// decays with a half-life afterwards, so one glitch fades but sustained evidence does not.
export const EVIDENCE_HALF_LIFE_MS = 120000;
const ACCRUAL = 0.35; // share of a signal's severity taken in per second of firing

export class TrustLedger {
  constructor({ halfLifeMs = EVIDENCE_HALF_LIFE_MS } = {}) {
    this.halfLifeMs = halfLifeMs;
    this.reset();
  }

  reset() {
    this.evidence = new Map(); // id -> { score, peak, firstT, lastT, seconds }
    this.last = null;
  }

  // Call once per tick with everything currently firing.
  observe(active, now) {
    const dt = this.last == null ? 0 : Math.max(0, now - this.last);
    this.last = now;
    const decay = dt ? 0.5 ** (dt / this.halfLifeMs) : 1;

    for (const e of this.evidence.values()) e.score *= decay;

    for (const { id, severity } of active) {
      if (!SIGNALS[id]) continue;
      const e = this.evidence.get(id) || { score: 0, peak: 0, firstT: now, lastT: now, seconds: 0 };
      // Noisy-or accrual: each second of firing adds evidence without ever exceeding certainty.
      const gain = severity * ACCRUAL * (dt ? Math.min(dt, 4000) / 1000 : 1);
      e.score = 1 - (1 - e.score) * (1 - Math.min(1, gain));
      e.peak = Math.max(e.peak, severity);
      e.lastT = now;
      e.seconds += dt / 1000;
      this.evidence.set(id, e);
    }
    for (const [id, e] of this.evidence) if (e.score < 0.005) this.evidence.delete(id);
  }

  // Standing signals (metadata, content) are facts about the call, so they apply at full weight directly.
  trust(standing = []) {
    let miss = 1;
    for (const [id, e] of this.evidence) miss *= 1 - SIGNALS[id].weight * e.score;
    for (const { id, severity } of standing) {
      if (SIGNALS[id]) miss *= 1 - SIGNALS[id].weight * severity;
    }
    return Math.max(0, Math.min(1, miss));
  }

  // Strongest accumulated evidence first, for explaining the score.
  summary() {
    return [...this.evidence.entries()]
      .map(([id, e]) => ({ id, ...e, contribution: SIGNALS[id].weight * e.score }))
      .sort((a, b) => b.contribution - a.contribution);
  }
}

export function trustLabel(trust) {
  if (trust >= 0.7) return { key: 'ok', text: 'Looks authentic' };
  if (trust >= 0.45) return { key: 'warn', text: 'Be cautious' };
  return { key: 'bad', text: 'High risk' };
}
