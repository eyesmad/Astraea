// WebSocket endpoint /ws/analyze on the HTTP port: the browser streams audio and face frames, the server
// streams back the evaluated signals once a second.
import { WebSocketServer } from 'ws';
import { AnalysisSession } from './analysis.js';

const SAMPLE_RATES = new Set([16000, 32000, 44100, 48000]);
const MAX_CHUNK_SAMPLES = 48000;

export function attachAnalysis(httpServer, port) {
  const wss = new WebSocketServer({ noServer: true, maxPayload: 1 << 20 });
  const allowed = new Set([`http://localhost:${port}`, `http://127.0.0.1:${port}`]);

  httpServer.on('upgrade', (req, socket, head) => {
    const { pathname } = new URL(req.url, 'http://x');
    if (pathname !== '/ws/analyze') { socket.destroy(); return; }
    // Browsers attach Origin to WebSocket handshakes; refuse other sites so a web page cannot drive this socket.
    if (req.headers.origin && !allowed.has(req.headers.origin)) {
      socket.write('HTTP/1.1 403 Forbidden\r\n\r\n');
      socket.destroy();
      return;
    }
    wss.handleUpgrade(req, socket, head, (ws) => wss.emit('connection', ws, req));
  });

  wss.on('connection', (ws) => {
    let session = new AnalysisSession();
    const send = (obj) => ws.readyState === ws.OPEN && ws.send(JSON.stringify(obj));
    const timer = setInterval(() => send({ type: 'signals', results: session.evaluate(), frames: session.frames }), 1000);

    ws.on('message', (data, isBinary) => {
      try {
        if (isBinary) {
          const buf = Buffer.isBuffer(data) ? data : Buffer.concat(data);
          const n = (buf.length - 8) >> 1;
          if (n <= 0 || n > MAX_CHUNK_SAMPLES) return;
          const samples = new Float32Array(n);
          for (let i = 0; i < n; i++) samples[i] = buf.readInt16LE(8 + 2 * i) / 32768;
          const endT = buf.readDoubleLE(0);
          if (Number.isFinite(endT)) session.pushAudio(samples, endT);
          return;
        }
        const msg = JSON.parse(data.toString());
        if (msg.type === 'config' && SAMPLE_RATES.has(msg.sampleRate)) {
          session = new AnalysisSession({ sampleRate: msg.sampleRate, fftSize: msg.sampleRate > 24000 ? 2048 : 1024 });
          send({ type: 'ready' });
        } else if (msg.type === 'face' && Number.isFinite(msg.t)) {
          session.pushFace(msg.t, msg.p);
        } else if (msg.type === 'reset') {
          session.reset();
        }
      } catch (err) {
        send({ type: 'error', message: err.message });
      }
    });
    ws.on('close', () => clearInterval(timer));
    ws.on('error', () => clearInterval(timer));
  });
  return wss;
}
