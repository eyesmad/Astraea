# Astraea Winning Hackathon Project

astraea AI

Real-time deepfake and scam-call detector (proof of concept web app, desktop browser).

## Run

```bash
npm install
python -m venv venv && venv/Scripts/pip install -r server/requirements.txt   # venv/bin/pip on macOS and Linux
cp .env.example .env   # optional: add IPQS_API_KEY / GEMINI_API_KEY
npm start
```

## Ports

| Port | What | Override |
|------|------|----------|
| 3000 | Web app and HTTP API (`http://localhost:3000`) | `PORT` |
| 3001 | WebSocket relay (`ws://localhost:3001`), rebroadcasts each message to the other clients | `WSS_PORT` |
| 3002 | Python analysis server, reached through the proxy rather than directly | - |

Both bind locally; the HTTP server listens on 127.0.0.1 only.

`npm start` runs both halves: the Node web server, and the Python analysis server on port 3002 that it proxies
`ws://localhost:3000/ws/analyze` to. During capture the browser streams 32 kHz PCM and compact face landmarks there; the
analysis server computes the voice heuristics (pitch, spectral cliff, breath coupling), face metrics (blink, jitter,
expression) and lip-sync lag, and returns signals once a second. MediaPipe runs in the browser because it needs the video
frames. If the socket cannot be reached the app falls back to analysing in the browser and says so in the overlay.

The heuristics exist twice: `server/analysis/` (Python) is what runs, and `public/js/*.js` is the reference the browser
fallback and the replay tooling use. Both are tested with the same expected values, so they cannot quietly drift apart.

## Test

Automated, covering both halves (`npm run test:js` and `npm run test:py` run them separately):

```bash
npm test
```

Manual:

1. `npm start`, open http://localhost:3000.
2. In the call card, pick a demo scenario (legit bank, grandparent scam, tech support scam) and check the number. Metadata signals appear in the overlay panel.
3. Paste the scenario transcript into the content box and classify. Without an Anthropic key an offline keyword classifier is used.
4. Use the "simulate" buttons to fire lip-sync and blink signals. Two or more live signals within about 10 s raise a compound alert.
5. For live analysis, start capture (webcam, or screen share with audio) and allow camera/mic access.

WebSocket relay: connect two clients to `ws://localhost:3001`; a message sent by one arrives at the other.

## What the overlay shows

Every face in the shared video gets a marker: solid brackets on the one the metrics follow (the largest), dashed brackets
labelled "not analysed" on the rest, so it is clear who the score is about when more than one person is on screen.

Lip-sync is judged against the device, not an absolute number. Every camera and microphone pair has its own
latency, so a genuine call routinely shows 100-300 ms of constant audio/video offset; the app learns that
baseline over the first few measurements and only flags movement away from it, a sync that wanders, or an
offset too large for any capture pipeline to explain. Losing lip/voice correspondence has to persist across
several readings before it counts, since a pause in speech briefly looks the same.

The trust score is cumulative. Evidence accumulates while a signal fires and decays with a two-minute half-life
afterwards, rather than only reflecting the current window: a caller who looked synthetic for twenty seconds should not
read as trustworthy the moment the signal stops. The line under the score says whether the evidence is current ("Now:")
or remembered ("Earlier:").

## Optional voice model

A voice-deepfake model can run alongside the heuristics as the `VOICE_MODEL` signal. It is off by default and
is never required: if Python or the weights are missing, the signal is simply absent.

```bash
venv/Scripts/pip install -r tools/requirements-voice-model.txt
VOICE_MODEL=1 npm start
```

It loads on a worker thread at first use, which takes a couple of minutes (the first run also downloads about 1.2 GB of
weights). It scores a 4-second window every few seconds off the event loop, so capture is never blocked, and fires only
when the median of at least 3 windows passes the threshold.

Measured on 210 clips, [pellav2](https://huggingface.co/Sadanie/pellav2-audio-deepfake-detector) scored 0.980
AUC on phone-quality audio, the best of the three models tested. Treat that number with suspicion: the upload
is unofficial, so it cannot be ruled out that the evaluation sets were in its training data. The model is also
confident on anything that is not speech (a sine tone and white noise both score over 99% synthetic), which is
why only windows containing voiced audio are scored. Calibrate the threshold on your own recordings before
trusting it.

## Record and replay

Thresholds are tuned against recorded calls rather than guesses. Add `?record=<label>` to the URL and start a
capture: the server writes everything the browser streams (audio chunks and face landmarks) to
`recordings/<label>-<timestamp>.ndjson`. Use `genuine` for a real person and `fake` for a known deepfake, since
the summary groups by those two labels.

```bash
node tools/replay.js recordings/genuine-2026-10-17.ndjson   # one capture, with a timeline
npm run replay -- --summary                                  # every recording, grouped by label
npm run replay -- --summary --json                           # machine-readable
```

Replay reconstructs the same analysis session the live socket built and drives it from the recorded clock, so
it needs no camera and always produces the same result. On a capture labelled `genuine`, every tick a signal
fires is a false positive; that percentage is the number to tune against. `recordings/` is gitignored, since
captures contain real faces and voices.

## Caveats

- Detection thresholds are untuned proof-of-concept values.
- Learning the lip-sync baseline assumes the first several seconds of a call are honest. A fake that is
  desynchronised from the very first frame is only caught once its offset exceeds the gross limit.
- STIR/SHAKEN cannot be read in a browser; it comes from the Twilio webhook (`POST /api/webhook/twilio`) or the demo scenarios.
