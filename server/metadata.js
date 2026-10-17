// Part 1: telephony and metadata signals.
// Each signal: { id, layer, label, fired, severity (0..1), detail }

import { parsePhoneNumberFromString } from 'libphonenumber-js/max';
import { getScenario } from './scenarios.js';

// Demo subset of North American area codes. Replace with a full NANP dataset for production.
const NANP_AREA = {
  '212': ['NY', 'New York'], '718': ['NY', 'New York'], '917': ['NY', 'New York'], '516': ['NY', 'New York'],
  '305': ['FL', 'Florida'], '786': ['FL', 'Florida'], '407': ['FL', 'Florida'],
  '213': ['CA', 'California'], '310': ['CA', 'California'], '415': ['CA', 'California'], '650': ['CA', 'California'],
  '408': ['CA', 'California'], '619': ['CA', 'California'],
  '214': ['TX', 'Texas'], '469': ['TX', 'Texas'], '713': ['TX', 'Texas'], '512': ['TX', 'Texas'],
  '312': ['IL', 'Illinois'], '773': ['IL', 'Illinois'],
  '202': ['DC', 'District of Columbia'], '206': ['WA', 'Washington'], '617': ['MA', 'Massachusetts'],
  '404': ['GA', 'Georgia'], '602': ['AZ', 'Arizona'], '702': ['NV', 'Nevada'],
};
const TOLL_FREE = new Set(['800', '833', '844', '855', '866', '877', '888']);

const norm = (s) => String(s || '').trim().toLowerCase();

export function areaCodeInfo(nationalNumber) {
  const npa = String(nationalNumber || '').slice(0, 3);
  if (TOLL_FREE.has(npa)) return { npa, tollFree: true, region: null };
  const hit = NANP_AREA[npa];
  return { npa, tollFree: false, region: hit ? { abbr: hit[0], name: hit[1] } : null };
}

// Accepts a Twilio-style StirVerstat value or a bare A/B/C.
export function parseVerstat(raw) {
  if (!raw || /^unknown$/i.test(raw)) return { status: 'unavailable', level: null };
  const s = String(raw).trim();
  if (/^No-TN-Validation$/i.test(s) || /^none$/i.test(s)) return { status: 'none', level: null };
  const m = /^(?:TN-Validation-)?(?:(Passed|Failed)-)?([ABC])$/i.exec(s);
  if (m) {
    const failed = (m[1] || '').toLowerCase() === 'failed';
    return { status: failed ? 'failed' : 'passed', level: m[2].toUpperCase() };
  }
  return { status: 'unavailable', level: null };
}

export function normalizeLineType(ipqs, libType) {
  const lt = norm(ipqs?.line_type);
  if (ipqs?.VOIP || lt.includes('voip') || libType === 'VOIP') {
    return ipqs?.prepaid ? 'disposable' : 'voip';
  }
  if (lt.includes('toll') || libType === 'TOLL_FREE') return 'tollfree';
  if (lt.includes('prepaid') || ipqs?.prepaid) return 'prepaid';
  if (lt.includes('wireless') || lt.includes('mobile') || libType === 'MOBILE') return 'mobile';
  if (lt.includes('landline') || libType === 'FIXED_LINE') return 'landline';
  if (libType === 'FIXED_LINE_OR_MOBILE') return 'landline_or_mobile';
  return 'unknown';
}

async function fetchIpqs(number, key) {
  const url = `https://ipqualityscore.com/api/json/phone/${encodeURIComponent(key)}/${encodeURIComponent(number)}`;
  const res = await fetch(url, { signal: AbortSignal.timeout(6000) });
  if (!res.ok) throw new Error(`IPQualityScore HTTP ${res.status}`);
  const j = await res.json();
  if (j.success === false) throw new Error(`IPQualityScore: ${j.message || 'request failed'}`);
  return j;
}

export async function reverseGeocode(lat, lon) {
  try {
    const url = `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${lon}&localityLanguage=en`;
    const res = await fetch(url, { signal: AbortSignal.timeout(4000) });
    if (!res.ok) return null;
    const j = await res.json();
    return { country: j.countryCode, region: j.principalSubdivision, city: j.city || j.locality };
  } catch {
    return null;
  }
}

const sig = (id, label, fired, severity, detail) => ({
  id, layer: 'metadata', label, fired, severity: fired ? severity : 0, detail,
});

export function buildSignals({ pn, verstat, intel, lineType, call, userLoc }) {
  const out = [];

  // CALLER_AUTH
  const v = parseVerstat(verstat);
  if (v.status === 'unavailable') {
    out.push(sig('CALLER_AUTH', 'STIR/SHAKEN', false, 0, 'No attestation supplied by the carrier or provider.'));
  } else if (v.status === 'failed') {
    out.push(sig('CALLER_AUTH', 'STIR/SHAKEN', true, 1, `Attestation ${v.level} FAILED validation. The caller ID was likely spoofed.`));
  } else if (v.status === 'none') {
    out.push(sig('CALLER_AUTH', 'STIR/SHAKEN', true, 0.45, 'Call carried no STIR/SHAKEN attestation.'));
  } else if (v.level === 'A') {
    out.push(sig('CALLER_AUTH', 'STIR/SHAKEN', false, 0, 'Attestation A: the carrier knows the caller and verified their right to this number.'));
  } else if (v.level === 'B') {
    out.push(sig('CALLER_AUTH', 'STIR/SHAKEN', true, 0.5, 'Attestation B: the carrier knows the caller but cannot verify the number.'));
  } else {
    out.push(sig('CALLER_AUTH', 'STIR/SHAKEN', true, 0.85, 'Attestation C (gateway): the carrier has no relationship with the caller. Common for international scam traffic.'));
  }

  // NUMBER_REPUTATION
  if (!intel) {
    out.push(sig('NUMBER_REPUTATION', 'Number reputation', false, 0, 'No reputation provider configured.'));
  } else {
    const score = Number(intel.fraud_score ?? 0);
    const flags = [intel.recent_abuse && 'recent abuse', intel.spammer && 'known spammer', intel.risky && 'flagged risky'].filter(Boolean);
    const fired = score >= 75 || flags.length >= 2;
    const sev = Math.min(1, Math.max(score / 100, flags.length ? 0.6 : 0));
    out.push(sig('NUMBER_REPUTATION', 'Number reputation', fired, sev,
      `Fraud score ${score}/100${flags.length ? ` (${flags.join(', ')})` : ''}.`));
  }

  // LINE_TYPE
  const lt = lineType;
  const claimsMobile = /mobile|cell|personal/i.test(call.claimedLineType || '');
  if (lt === 'disposable') {
    out.push(sig('LINE_TYPE', 'Line type', true, 0.8, 'Disposable or virtual number (prepaid VoIP). Heavily used by scam operations.'));
  } else if (lt === 'voip') {
    out.push(sig('LINE_TYPE', 'Line type', true, claimsMobile ? 0.8 : 0.55,
      claimsMobile ? 'Claims to be a personal mobile but resolves to a VoIP provider.' : 'VoIP number. Cheap to obtain and easy to rotate.'));
  } else if (lt === 'tollfree' || lt === 'prepaid') {
    out.push(sig('LINE_TYPE', 'Line type', true, 0.35, `${lt === 'tollfree' ? 'Toll-free' : 'Prepaid'} line. Inbound-only or low accountability.`));
  } else {
    out.push(sig('LINE_TYPE', 'Line type', false, 0, `Line type: ${lt}.`));
  }

  // AREA_CODE: neighbor spoofing shows an area code that differs from where the carrier registered the number.
  const ac = pn?.countryCallingCode === '1' ? areaCodeInfo(pn.nationalNumber) : null;
  const regRegion = intel?.region;
  if (ac?.region && regRegion) {
    const same = norm(regRegion) === norm(ac.region.name) || norm(regRegion) === norm(ac.region.abbr);
    out.push(sig('AREA_CODE', 'Area code', !same, 0.6,
      same ? `Area code ${ac.npa} matches the carrier's registered region (${regRegion}).`
        : `Area code ${ac.npa} is ${ac.region.name}, but the carrier registered this number in ${regRegion}. Possible neighbor spoofing.`));
  } else {
    out.push(sig('AREA_CODE', 'Area code', false, 0, ac?.tollFree ? `Toll-free prefix ${ac.npa}; no geography.` : 'Area code not checkable.'));
  }

  // GEO_MISMATCH
  const numCountry = intel?.country || pn?.country || null;
  const numRegion = intel?.region || null;
  const origin = call.originCountry || null;
  const userCountry = userLoc?.country || null;
  const userRegion = userLoc?.region || null;
  const issues = [];
  let sev = 0;
  if (numCountry && origin && norm(numCountry) !== norm(origin)) {
    issues.push(`number is registered in ${numCountry} but the call entered the network from ${origin}`);
    sev = Math.max(sev, 0.9);
  }
  if (numCountry && userCountry && norm(numCountry) !== norm(userCountry)) {
    issues.push(`number is registered in ${numCountry} but you are in ${userCountry}`);
    sev = Math.max(sev, call.claimedLocal ? 0.8 : 0.4);
  }
  if (call.claimedLocal && numRegion && userRegion && norm(numRegion) !== norm(userRegion)) {
    issues.push(`claims to be local but the number is registered in ${numRegion}, not ${userRegion}`);
    sev = Math.max(sev, 0.7);
  }
  if (!numCountry && !origin && !userCountry) {
    out.push(sig('GEO_MISMATCH', 'Geography', false, 0, 'Not enough location data to compare.'));
  } else {
    out.push(sig('GEO_MISMATCH', 'Geography', issues.length > 0, sev,
      issues.length ? `${issues.join('; ')}.` : 'Number, network origin and your location are consistent.'));
  }

  return out;
}

function stripEmpty(o) {
  const r = {};
  for (const [k, v] of Object.entries(o || {})) if (v !== '' && v != null && k !== 'demo') r[k] = v;
  return r;
}

export async function analyzeMetadata(input, env = process.env) {
  const scenario = input.demo ? getScenario(input.demo) : null;
  const call = { ...(scenario?.call || {}), ...stripEmpty(input) };
  if (scenario) Object.assign(call, scenario.call);
  if (!call.number) throw Object.assign(new Error('A phone number is required.'), { status: 400 });

  const pn = parsePhoneNumberFromString(String(call.number), 'US');
  if (!pn) throw Object.assign(new Error('Could not parse that phone number.'), { status: 400 });

  let intel = null;
  let source = 'none';
  if (scenario) {
    intel = scenario.ipqs;
    source = 'demo';
  } else if (env.IPQS_API_KEY) {
    intel = await fetchIpqs(pn.number, env.IPQS_API_KEY);
    source = 'ipqualityscore';
  }

  let userLoc = call.user || {};
  if (!userLoc.country && call.lat != null && call.lon != null) {
    userLoc = (await reverseGeocode(call.lat, call.lon)) || userLoc;
  }

  const lineType = normalizeLineType(intel, pn.getType?.());
  const signals = buildSignals({ pn, verstat: call.verstat, intel, lineType, call, userLoc });

  return {
    source,
    number: pn.number,
    country: pn.country || null,
    lineType,
    carrier: intel?.carrier || null,
    registered: intel ? { country: intel.country, region: intel.region, city: intel.city } : null,
    user: userLoc,
    originCountry: call.originCountry || null,
    claimedOrg: call.claimedOrg || '',
    signals,
    transcript: scenario?.transcript || null,
  };
}
