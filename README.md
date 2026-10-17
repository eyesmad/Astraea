# Astraea Winning Hackathon Project Inshallah

astraea AI, powered by allah

Real-time deepfake and scam-call detector (proof of concept web app, desktop browser).

## Run

```bash
npm install
cp .env.example .env   # optional: add IPQS_API_KEY / GEMINI_API_KEY
npm start
```

## Ports

| Port | What | Override |
|------|------|----------|
| 3000 | Web app and HTTP API (`http://localhost:3000`) | `PORT` |
| 3001 | WebSocket relay (`ws://localhost:3001`), rebroadcasts each message to the other clients | `WSS_PORT` |

Both bind locally; the HTTP server listens on 127.0.0.1 only.

The HTTP port also serves `ws://localhost:3000/ws/analyze` (same-origin only). During capture the browser streams 32 kHz PCM
and compact face landmarks there; the server computes the audio heuristics (pitch, spectral cliff, breath coupling), face
metrics (blink, jitter, expression) and lip-sync lag, and returns signals once a second. MediaPipe itself still runs in the
browser because it needs the video frames. If the socket cannot be reached the app falls back to analysing in the browser.

## Test

Automated (unit tests, including the server-side analysis session and its WebSocket):

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

## Optional voice model

A voice-deepfake model can run alongside the heuristics as the `VOICE_MODEL` signal. It is off by default and
is never required: if Python or the weights are missing, the signal is simply absent.

```bash
python -m venv .venv-voice && .venv-voice/Scripts/pip install -r tools/requirements-voice-model.txt
VOICE_MODEL=1 VOICE_MODEL_PYTHON=.venv-voice/Scripts/python npm start
```

The server starts the model at boot, since loading it takes a couple of minutes (the first run also downloads
about 1.2 GB of weights). It scores a 4-second window every few seconds on a background process, so capture is
never blocked, and fires only when the median of at least 3 windows passes the threshold.

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

Set `VOICE_MODEL=1` on a replay too, to measure the model against the same recordings.

Replay reconstructs the same analysis session the live socket built and drives it from the recorded clock, so
it needs no camera and always produces the same result. On a capture labelled `genuine`, every tick a signal
fires is a false positive; that percentage is the number to tune against. `recordings/` is gitignored, since
captures contain real faces and voices.

## Caveats

- Detection thresholds are untuned proof-of-concept values.
- STIR/SHAKEN cannot be read in a browser; it comes from the Twilio webhook (`POST /api/webhook/twilio`) or the demo scenarios.
- Speech-to-text only hears the microphone, not the caller.
- Content classification runs through Gemini when `GEMINI_API_KEY` is set, and falls back to the offline keyword classifier otherwise or on an API failure.
