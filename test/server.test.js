import test from 'node:test';
import assert from 'node:assert/strict';
import { analyzeMetadata, parseVerstat, areaCodeInfo } from '../server/metadata.js';
import { classifyHeuristic, classify, normalizeResult } from '../server/scam.js';
import { SCENARIOS } from '../server/scenarios.js';

const byId = (r) => Object.fromEntries(r.signals.map((s) => [s.id, s]));

test('parseVerstat understands Twilio values and bare letters', () => {
  assert.deepEqual(parseVerstat('TN-Validation-Passed-A'), { status: 'passed', level: 'A' });
  assert.deepEqual(parseVerstat('TN-Validation-Failed-C'), { status: 'failed', level: 'C' });
  assert.deepEqual(parseVerstat('B'), { status: 'passed', level: 'B' });
  assert.equal(parseVerstat('No-TN-Validation').status, 'none');
  assert.equal(parseVerstat('').status, 'unavailable');
});

test('areaCodeInfo', () => {
  assert.equal(areaCodeInfo('3055550187').region.name, 'Florida');
  assert.equal(areaCodeInfo('8885550133').tollFree, true);
});

test('legit bank scenario fires no metadata signals', async () => {
  const r = byId(await analyzeMetadata({ demo: 'legit-bank' }, {}));
  assert.deepEqual(Object.values(r).filter((s) => s.fired).map((s) => s.id), []);
});

test('grandparent scenario fires every metadata signal', async () => {
  const r = byId(await analyzeMetadata({ demo: 'grandparent' }, {}));
  for (const id of ['CALLER_AUTH', 'NUMBER_REPUTATION', 'GEO_MISMATCH', 'LINE_TYPE', 'AREA_CODE']) {
    assert.equal(r[id].fired, true, `${id} should fire`);
  }
  assert.ok(r.LINE_TYPE.detail.toLowerCase().includes('disposable'));
  assert.ok(r.GEO_MISMATCH.detail.includes('NG'));
  assert.ok(r.GEO_MISMATCH.detail.includes('Florida'));
});

test('tech support scenario flags attestation B, VoIP and toll-free geography', async () => {
  const r = byId(await analyzeMetadata({ demo: 'tech-support' }, {}));
  assert.equal(r.CALLER_AUTH.severity, 0.5);
  assert.equal(r.LINE_TYPE.fired, true);
  assert.equal(r.GEO_MISMATCH.fired, true);
});

test('custom number without a reputation key still checks attestation and line type', async () => {
  const r = await analyzeMetadata({ number: '+1 415 555 0100', verstat: 'C' }, {});
  assert.equal(r.source, 'none');
  assert.equal(byId(r).CALLER_AUTH.fired, true);
});

test('unparseable number is rejected', async () => {
  await assert.rejects(() => analyzeMetadata({ number: 'abc' }, {}), /parse|required/i);
});

test('heuristic classifier separates scams from the legitimate call', () => {
  const legit = classifyHeuristic(SCENARIOS[0].transcript);
  const grand = classifyHeuristic(SCENARIOS[1].transcript);
  const tech = classifyHeuristic(SCENARIOS[2].transcript);
  assert.equal(legit.category, 'none');
  assert.equal(grand.category, 'grandparent');
  assert.equal(tech.category, 'tech_support');
  assert.ok(grand.red_flags.some((f) => f.tactic === 'payment_method'));
  assert.ok(grand.red_flags.some((f) => f.tactic === 'secrecy'));
});

test('classify falls back offline without an API key and rejects empty input', async () => {
  const r = await classify(SCENARIOS[1].transcript, 'Grandchild', {});
  assert.equal(r.engine, 'heuristic');
  await assert.rejects(() => classify('  ', '', {}), /empty/i);
});

test('normalizeResult clamps model output into the schema', () => {
  const r = normalizeResult({ category: 'made_up', confidence: 7, red_flags: [{ tactic: 'urgency' }] }, 'gemini');
  assert.equal(r.category, 'other_scam');
  assert.equal(r.confidence, 1);
  assert.equal(r.red_flags[0].quote, '');
  assert.equal(r.engine, 'gemini');
});

test('normalizeResult survives junk', () => {
  const r = normalizeResult({ confidence: 'abc', red_flags: ['urgency', null], recommended_actions: 'nope' }, 'gemini');
  assert.equal(r.confidence, 0.5);
  assert.deepEqual(r.red_flags, [{ tactic: 'urgency', quote: '', why: '' }]);
  assert.deepEqual(r.recommended_actions, []);
  assert.equal(normalizeResult(null, 'gemini').category, 'other_scam');
});

// Regression: classifyWithGemini used to swallow a failed request and return undefined, which made the
// fallback below unreachable and sent an empty body to the client.
test('classify falls back to the offline classifier when Gemini fails', async (t) => {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    calls.push(url);
    return new Response('upstream exploded', { status: 500 });
  });
  const r = await classify('You must buy gift cards right now or you will be arrested.', '', { GEMINI_API_KEY: 'x' });
  assert.equal(calls.length, 1);
  assert.equal(r.engine, 'heuristic');
  assert.match(r.note, /Gemini unavailable/);
  assert.ok(r.red_flags.length > 0);
});

test('classify normalizes a successful Gemini response', async (t) => {
  const body = { category: 'tech_support', confidence: 0.91, summary: 'Caller wants remote access.', red_flags: [{ tactic: 'remote_access', quote: 'install AnyDesk', why: 'Wants control of the device.' }], recommended_actions: ['Hang up.'] };
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ candidates: [{ content: { parts: [{ text: JSON.stringify(body) }] } }] }), { status: 200 }));
  const r = await classify('Please install AnyDesk so I can fix your computer.', '', { GEMINI_API_KEY: 'x' });
  assert.equal(r.engine, 'gemini');
  assert.equal(r.category, 'tech_support');
  assert.equal(r.red_flags[0].tactic, 'remote_access');
  assert.equal(r.note, undefined);
});

test('classify falls back when Gemini returns no content', async (t) => {
  t.mock.method(globalThis, 'fetch', async () => new Response(JSON.stringify({ candidates: [] }), { status: 200 }));
  const r = await classify('Hello, this is your bank.', '', { GEMINI_API_KEY: 'x' });
  assert.equal(r.engine, 'heuristic');
  assert.match(r.note, /no content/);
});
