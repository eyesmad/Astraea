import express from 'express';
import WebSocket, { WebSocketServer } from 'ws';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzeMetadata } from './server/metadata.js';
import { classify } from './server/scam.js';
import { SCENARIOS } from './server/scenarios.js';
import { attachAnalysis } from './server/analysisSocket.js';
import { sharedVoiceModel, voiceModelEnabled } from './server/voiceModel.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Minimal .env loader so no extra dependency is needed.
const envFile = path.join(__dirname, '.env');
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
    const m = /^\s*([A-Z0-9_]+)\s*=\s*(.*?)\s*$/.exec(line);
    if (m && m[2] && !(m[1] in process.env)) process.env[m[1]] = m[2].replace(/^["']|["']$/g, '');
  }
}

const app = express();
app.use(express.json({ limit: '256kb' }));
app.use(express.urlencoded({ extended: false }));
app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/config', (_req, res) => {
  res.json({
    ipqs: Boolean(process.env.IPQS_API_KEY),
    llm: Boolean(process.env.GEMINI_API_KEY),
    scenarios: SCENARIOS.map((s) => ({ id: s.id, label: s.label, call: s.call })),
  });
});

const wrap = (fn) => async (req, res) => {
  try {
    res.json(await fn(req));
  } catch (err) {
    res.status(err.status || 500).json({ error: err.message });
  }
};

app.post('/api/metadata', wrap((req) => analyzeMetadata(req.body || {})));
app.post('/api/classify', wrap((req) => classify(req.body?.transcript, req.body?.claimedOrg)));

// Twilio-style voice webhook: carries StirVerstat when the number has STIR/SHAKEN data.
// For production, validate X-Twilio-Signature before trusting this.
let latestCall = null;
app.post('/api/webhook/twilio', (req, res) => {
  const b = req.body || {};
  latestCall = {
    number: b.From || '',
    verstat: b.StirVerstat || '',
    originCountry: b.CallerCountry || b.FromCountry || '',
    callSid: b.CallSid || '',
    receivedAt: Date.now(),
  };
  res.type('text/xml').send('<?xml version="1.0" encoding="UTF-8"?><Response/>');
});
app.get('/api/calls/latest', (_req, res) => res.json(latestCall));

const port = Number(process.env.PORT) || 3000;
const wssPort = Number(process.env.WSS_PORT) || 3001;

// WebSocket relay on its own port: rebroadcasts each message to the other clients.
const wss = new WebSocketServer({ port: wssPort });
wss.on('connection', (socket) => {
  console.log(`WS client connected. Total: ${wss.clients.size}`);
  socket.on('message', (data, isBinary) => {
    for (const client of wss.clients) {
      if (client !== socket && client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
    }
  });
  socket.on('close', () => console.log(`WS client disconnected. Total: ${wss.clients.size}`));
  socket.on('error', (err) => console.error(`Socket error: ${err.message}`));
});

const httpServer = app.listen(port, '127.0.0.1', () => {
  console.log(`Astraea running at http://localhost:${port} (WebSocket on ws://localhost:${wssPort})`);
});
attachAnalysis(httpServer, port);

// Loading the weights takes minutes, so start the sidecar at boot rather than during the first call.
if (voiceModelEnabled()) {
  console.log('Voice model enabled, loading weights (first run downloads about 1.2 GB)...');
  sharedVoiceModel().start().then((ok) => console.log(ok ? 'Voice model ready.' : 'Voice model unavailable; using voice heuristics only.'));
}
