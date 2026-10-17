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

export function trustLabel(trust) {
  if (trust >= 0.7) return { key: 'ok', text: 'Looks authentic' };
  if (trust >= 0.45) return { key: 'warn', text: 'Be cautious' };
  return { key: 'bad', text: 'High risk' };
}
