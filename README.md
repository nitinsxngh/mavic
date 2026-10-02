# Mavic · Video Agent Buddy

Camera is the eyes. Microphone is the ears. LangGraph is the brain.

This is the V1 implementation of that split: a real-time perception pipeline that **selectively** produces visual and audio context, then feeds a compact `AgentContext` into LangGraph. Continuous video never enters the agent.

**Full technical document (architecture, turn loop, APIs, LangGraph, security):** [docs/TECHNICAL.md](docs/TECHNICAL.md)

## What V1 can do

1. Talk with you over camera + microphone (OpenAI Realtime via WebRTC).
2. Retrieve personal facts from PostgreSQL + pgvector, always filtered by `user_id`.
3. Report your current location from browser GPS, then reverse-geocode it. Location is never inferred by the LLM.

Emotion from voice is treated as **probabilistic context**, not truth: “you may sound tired”, never “you are tired”.

## Architecture

```
Browser (camera / mic / GPS / speaker)
        │  WebRTC audio + sampled JPEG frames
        ▼
FastAPI gateway
        │
        ├── perception: VAD-triggered 3–5 frame window, audio features, GPS
        ├── context assembler
        └── LangGraph (intent → parallel tools → one response)
                    ├── Memory agent  → pgvector (user namespace)
                    ├── Vision agent  → multi-frame verification
                    └── Location agent → GPS + Nominatim
```

The realtime voice model is the nervous system (speech in/out, tool calls). LangGraph and the tool implementations are the cognitive layer. `user_id` is injected by the backend, never chosen by the model.

## Stack

| Layer | V1 choice |
| --- | --- |
| Frontend | Next.js, MediaDevices, Geolocation, WebRTC |
| Voice | `gpt-realtime-2.1` over WebRTC |
| Agent | Python, FastAPI, LangGraph |
| Memory | PostgreSQL + pgvector |
| Session | Redis (in-memory fallback if Redis is down) |
| Vision | Selected JPEGs → vision chat model |

## Run locally

1. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
2. Local V1 does **not** need Docker. It uses SQLite for memory and an in-process session store if Redis is down.

3. Backend:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python scripts/seed_memory.py
uvicorn app.main:app --reload --port 8000
```

4. Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000), allow camera, microphone, and location, then talk.

Try:

- “What is my name?”
- “Where am I right now?”
- “What am I holding?”
- “Remember that I prefer dark mode.”

Text input still runs the LangGraph turn path if you do not want to speak.

## Design rules baked into the code

- Do not send 30 FPS into LangGraph. Speech start captures a small frame window (`t-buffer`, `t0`, `t+200ms`, `t+500ms`) and drops duplicates.
- Vision is verified across frames before the agent trusts an object/color.
- Memory search always includes `WHERE user_id = authenticated_user`.
- GPS comes from the browser. IP location is not used.
- Tools have permission labels (`private_read`, `sensitive_read`, `private_write`) for later authz.
- Perceived latency: the voice layer can say “Let me take a look” while vision runs.

## What is deliberately not in V1

Kubernetes, Kafka, a separate microservice per sense, proactive “I noticed a new person” triggers, and storing every camera frame. Those belong in V2/V3 once this loop is solid.
