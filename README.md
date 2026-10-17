# Astraea Winning Hackathon Project Inshallah

astraea AI, powered by allah

Real-time deepfake and scam-call detector (proof of concept web app, desktop browser).

## Run

```bash
npm install
cp .env.example .env   # optional: add IPQS_API_KEY / ANTHROPIC_API_KEY
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

## Caveats

- Detection thresholds are untuned proof-of-concept values.
- STIR/SHAKEN cannot be read in a browser; it comes from the Twilio webhook (`POST /api/webhook/twilio`) or the demo scenarios.
- Speech-to-text only hears the microphone, not the caller.
