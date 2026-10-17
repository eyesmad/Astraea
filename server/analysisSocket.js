// WebSocket endpoint /ws/analyze on the HTTP port: the browser streams audio and face frames, the server
// streams back the evaluated signals once a second.
import { WebSocketServer } from 'ws';
import { SAMPLE_RATES, sessionFor } from './analysis.js';
import { RecordingWriter, decodeAudioChunk, recordingPath } from './recording.js';

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
    let session = sessionFor(32000);
    let recorder = null;
    const send = (obj) => ws.readyState === ws.OPEN && ws.send(JSON.stringify(obj));
    const timer = setInterval(() => send({ type: 'signals', results: session.evaluate(), probes: session.probes(), frames: session.frames }), 1000);
    const stop = () => {
      clearInterval(timer);
      recorder?.close();
      recorder = null;
    };

    ws.on('message', (data, isBinary) => {
      try {
        if (isBinary) {
          const buf = Buffer.isBuffer(data) ? data : Buffer.concat(data);
          const n = (buf.length - 8) >> 1;
          if (n <= 0 || n > MAX_CHUNK_SAMPLES) return;
          const chunk = decodeAudioChunk(buf);
          if (!chunk) return;
          session.pushAudio(chunk.samples, chunk.endT);
          recorder?.audio(buf);
          return;
        }
        const msg = JSON.parse(data.toString());
        if (msg.type === 'config' && SAMPLE_RATES.has(msg.sampleRate)) {
          session = sessionFor(msg.sampleRate);
          recorder?.close();
          recorder = null;
          if (msg.record) {
            const file = recordingPath(msg.record);
            recorder = new RecordingWriter(file, { sampleRate: msg.sampleRate, label: msg.record, note: typeof msg.note === 'string' ? msg.note.slice(0, 500) : '' });
            console.log(`Recording capture to ${file}`);
          }
          send({ type: 'ready', recording: Boolean(recorder) });
        } else if (msg.type === 'faces' && Number.isFinite(msg.t)) {
          const pFace = msg.faces.find(f => f.primary) || msg.faces[0];
          if (pFace && session.pushFace(msg.t, pFace.p, msg.mouth)) recorder?.face(msg.t, pFace.p);
        } else if (msg.type === 'reset') {
          session.reset();
        }
      } catch (err) {
        send({ type: 'error', message: err.message });
      }
    });
    ws.on('close', stop);
    ws.on('error', stop);
  });
  return wss;
}
