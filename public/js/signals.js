// Signal registry. `weight` is how much a fully-severe signal reduces trust.
// `standing` signals fire once per call (metadata, content) and stay until reset.
// Live signals are re-reported while they hold and expire after LIVE_TTL_MS.

export const LAYERS = {
  metadata: { label: 'Caller metadata', question: 'Is this number who it claims to be?' },
  video: { label: 'Video', question: 'Does the face behave like a real one?' },
  voice: { label: 'Voice', question: 'Does the voice behave like a real one?' },
  crossmodal: { label: 'Lips and voice', question: 'Do the lips match the sound?' },
  content: { label: 'What is said', question: 'Does the call follow a known scam script?' },
  challenge: { label: 'Active challenge', question: 'Can the caller follow a live instruction?' },
};

export const SIGNALS = {
  CALLER_AUTH: { layer: 'metadata', label: 'STIR/SHAKEN attestation', weight: 0.7, standing: true },
  NUMBER_REPUTATION: { layer: 'metadata', label: 'Number reputation', weight: 0.6, standing: true },
  GEO_MISMATCH: { layer: 'metadata', label: 'Geographic mismatch', weight: 0.6, standing: true },
  LINE_TYPE: { layer: 'metadata', label: 'Line type', weight: 0.4, standing: true },
  AREA_CODE: { layer: 'metadata', label: 'Area code', weight: 0.4, standing: true },

  BLINK_CADENCE: { layer: 'video', label: 'Blink cadence', weight: 0.5 },
  MOTION_SMOOTHNESS: { layer: 'video', label: 'Motion smoothness', weight: 0.4 },
  EXPRESSION_FLAT: { layer: 'video', label: 'Expression variance', weight: 0.3 },

  VOCODER_ARTIFACT: { layer: 'voice', label: 'Vocoder signature', weight: 0.5 },
  PROSODY_FLAT: { layer: 'voice', label: 'Prosody', weight: 0.4 },
  BREATH_ABSENT: { layer: 'voice', label: 'Respiratory coupling', weight: 0.4 },
  VOICE_MODEL: { layer: 'voice', label: 'Voice model', weight: 0.6 },

  LIP_DRIFT: { layer: 'crossmodal', label: 'Lip-sync drift', weight: 0.7 },
  LIP_DECOUPLED: { layer: 'crossmodal', label: 'Lips decoupled from voice', weight: 0.5 },

  SCAM_CONTENT: { layer: 'content', label: 'Scam script match', weight: 0.7, standing: true },

  CHALLENGE_FAILED: { layer: 'challenge', label: 'Challenge failed', weight: 0.9, standing: true },
};

export const LIVE_TTL_MS = 12000;

// Pairs that are stronger together: a face swap that gets the lips wrong often gets blinks wrong too.
export const SYNERGY = [
  ['LIP_DRIFT', 'BLINK_CADENCE'],
  ['LIP_DRIFT', 'MOTION_SMOOTHNESS'],
  ['VOCODER_ARTIFACT', 'LIP_DRIFT'],
  ['PROSODY_FLAT', 'BREATH_ABSENT'],
  ['VOICE_MODEL', 'LIP_DRIFT'],
];
