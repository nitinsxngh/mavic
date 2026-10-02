# Mavic Agent — Technical Document

**Product:** Video Agent Buddy  
**Version:** V1 (local, as implemented)  
**Principle:** Perception is separate from reasoning. The camera is eyes, the microphone is ears, LangGraph is the brain. Continuous video never enters the LLM.

This document describes how the running system works, not a future Kubernetes design.

---

## 1. What V1 does

| Capability | How it is implemented |
| --- | --- |
| Talk about anything using camera + mic | Browser captures media. OpenAI Realtime (WebRTC) handles speech in/out. LangGraph generates the reply. |
| Personal memory | SQLite stores facts + embeddings (`personal_memories`). Retrieval is always filtered by `user_id`. |
| Named entities | SQLite `known_entities` remembers people, personal objects, and animals by name + visual description. Same `user_id` filter. |
| Current location | Browser Geolocation API → session store → Nominatim reverse geocode. The LLM never invents GPS. |
| Emotion | Web Audio features (energy, rate, pitch variance) mapped to *probabilistic* labels: “may sound tired”, never “you are tired”. |
| Language | Hindi (Devanagari) and English only. Urdu/Arabic script is forbidden in prompts and transcription config. |

---

## 2. High-level architecture

```
                         VIDEO AGENT BUDDY (browser)
                    Camera = eyes · Mic = ears · Speaker = voice
                    GPS = location · Web Audio = emotion signals
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
           WebRTC audio I/O                      JPEG frames (3–5)
           OpenAI Realtime                       POST /api/context
           (nervous system)                             │
                    │                                   │
                    │  transcript                       │
                    └─────────────────┬─────────────────┘
                                      ▼
                            FastAPI  :8000
                            Context assembler
                                      ▼
                            LangGraph (brain)
           ┌──────────┬──────────┬──────────┬──────────┐
           ▼          ▼          ▼          ▼          ▼
        Vision     Facts     Entities    Location   Reply text
        (frames)   memory    (named)     (GPS+geo)     │
        SQLite     SQLite    SQLite                     ▼
                                              Realtime speak()
                                              (speaker)
```

**Do not send 30 FPS into LangGraph.** Speech triggers a small set of JPEGs. Those images plus transcript, GPS, and emotion become one `AgentContext` object. Named people / objects / animals are stored and recalled after vision, not by streaming faces into a separate ID model.

---

## 3. Separation of perception and reasoning

### Perception layer (senses)

Answers: *What am I seeing / hearing / where am I?*

It produces structured JSON, for example:

```json
{
  "speech": "ये कौन है?",
  "visual": {
    "scene": "person holding a smartphone toward the camera",
    "objects": [{ "label": "smartphone", "color": "black", "confidence": 0.86 }],
    "verified_from_frames": 3
  },
  "audio": {
    "emotion": { "label": "neutral", "confidence": 0.45, "phrasing": "user MAY sound…" }
  },
  "location": { "latitude": 28.56, "longitude": 77.39, "accuracy_m": 12 }
}
```

### Cognitive layer (LangGraph)

Answers: *Given that context, what should I say, retrieve, or remember?*

The voice model is **not** the brain in V1. It transcribes speech and later speaks LangGraph’s text. Tool use (vision, facts memory, named entities, GPS) runs in Python.

---

## 4. Repository layout

```
Mavic Agent/
  frontend/                 Next.js 15 UI
    app/                    layout, page, CSS
    components/AgentStudio.tsx
    lib/                    api, frames, audio, location, realtime
  backend/
    app/
      main.py               FastAPI app, CORS, DB init
      api/routes.py         HTTP + WebRTC SDP
      agent/graph.py        LangGraph (intent, gather, reason)
      agent/tools.py        vision / memory / entities / location
      agent/language.py     Hindi Devanagari + English policy
      agent/state.py        AgentState
      perception/           frames, vision, emotion, geocode
      context/              AgentContext models + assembler
      memory/
        store.py            personal_memories + profile
        entities.py         known_entities extract / upsert / search
        embeddings.py       text-embedding-3-small
      db/models.py          SQLite (or Postgres+pgvector)
      session/store.py      Redis or in-memory
      security/auth.py      injected user_id
    scripts/seed_memory.py
  docs/TECHNICAL.md
  .env                      OPENAI_API_KEY, DATABASE_URL, …
```

---

## 5. Runtime stack (what actually runs locally)

| Layer | Implementation |
| --- | --- |
| UI | Next.js 15, React, TypeScript |
| API proxy | Next rewrites `/api/*` → `http://127.0.0.1:8000/api/*` (same-origin, avoids CORS) |
| Backend | FastAPI + Uvicorn, Python 3.9 |
| Voice | OpenAI Realtime `gpt-realtime-2.1` over WebRTC |
| Chat / vision | `gpt-4.1-mini` (configurable) |
| Embeddings | `text-embedding-3-small` |
| Memory DB | SQLite `backend/mavic.db` — `personal_memories`, `known_entities`, `user_profiles` (cosine in Python) |
| Session | In-memory dict if Redis is down |
| Location | `navigator.geolocation` + OpenStreetMap Nominatim |

Postgres + pgvector and Redis remain supported via `DATABASE_URL` / `REDIS_URL` when Docker is available. They are not required for V1.

`init_db()` runs `Base.metadata.create_all` on Uvicorn startup, so new tables (including `known_entities`) appear on the next backend start.

---

## 6. End-to-end turn (voice)

This is the loop after **Enable camera, mic, location**.

```
T+0     User starts speaking
        Realtime: input_audio_buffer.speech_started
        Frontend grabs 3 fresh JPEGs (~140ms apart) → POST /api/context

T+…     User stops
        Realtime transcribes (language=hi, Devanagari + English)

T+…     conversation.item.input_audio_transcription.completed
        Chat shows the user bubble
        Frontend waits for the in-flight capture, then captures again
        (so a phone raised mid-sentence is in the last frames)
        POST /api/turn { session_id, transcript, location, speech_features }

T+…     LangGraph:
          normalize → detect_intent
          → gather (vision ∥ facts memory ∥ location, then entity save/recall)
          → agent_reason → response_prepare

T+…     Chat shows assistant bubble
        Frontend sends Realtime response.create
        “Speak this reply. Do not call tools.”
        Speaker plays audio
```

If a second utterance arrives while a turn is running, it is queued and processed next. Turns are not dropped.

Typed **Send** uses the same `/api/turn` path, then the same `speak()`.

---

## 7. Frontend

### 7.1 Device APIs

- Camera + mic: `navigator.mediaDevices.getUserMedia({ audio, video })`
- Location: `navigator.geolocation.getCurrentPosition` (GPS, not IP)
- Speaker: `<audio autoPlay playsInline>` attached to the WebRTC remote stream

The camera preview is mirrored (`scaleX(-1)`). Frames sent to the backend are unmirrored canvas captures of the video element.

### 7.2 Frame sampler (`frontend/lib/frames.ts`)

- Ring buffer of recent JPEGs (640px wide, quality 0.72)
- Idle: grab about every 1.4s (pre-buffer)
- On speech / turn: `captureNow()` takes three frames at 0 / 140 / 320 ms, plus last ring frames, then drops near-duplicates
- **Never** 30 FPS into the model

### 7.3 Audio features (`frontend/lib/audio.ts`)

Web Audio AnalyserNode → `{ energy, speech_rate, pitch_variance }` in `[0, 1]`.

Backend maps those to labels such as `possibly_tired` with a confidence and the phrasing *user MAY sound…*.

### 7.4 Realtime client (`frontend/lib/realtime.ts`)

1. `POST /api/realtime/sdp` with the browser SDP offer (API key stays on the server)
2. Set remote SDP answer
3. Data channel `oai-events` for events
4. Recvonly audio transceiver so the model’s voice is heard
5. `speak(text)` → `response.create` with `output_modalities: ["audio"]`

Realtime session config (server-side):

- Transcription model `gpt-4o-mini-transcribe`, `language: "hi"`
- Prompt: Hindi in Devanagari, English in Latin, never Urdu script
- `turn_detection.create_response = false` so Realtime does **not** answer by itself
- **No tools** on the Realtime session (avoids the model waiting forever on function calls)

### 7.5 UI (`AgentStudio.tsx`)

Full-viewport studio: camera left, scrolling transcript right, live caption on the video, status chips (Listening / Looking / Thinking / Speaking), GPS chip, emotion chip.

---

## 8. Backend HTTP API

Base: `http://127.0.0.1:8000` (browser uses `/api` via Next rewrite).

| Method | Path | Role |
| --- | --- | --- |
| GET | `/health` | Liveness |
| POST | `/api/session` | Creates `session_id`, binds demo `user_id` |
| POST | `/api/context` | Stores frames, GPS, speech features, transcript in session |
| POST | `/api/turn` | Runs LangGraph; see response fields below |
| POST | `/api/tools/execute` | Direct tool call (legacy / debug) |
| POST | `/api/realtime/sdp` | Unified WebRTC handshake with OpenAI |

`POST /api/turn` returns:

`ack`, `intent`, `required_tools`, `response`, `vision`, `location`, `memories`, `entities`, `saved_entity`, `emotion`, `frames_buffered`.

Identity: in `APP_ENV=development`, `user_id` is `DEFAULT_USER_ID` (`usr_demo`). The LLM never chooses `user_id`.

---

## 9. Context assembler

`assemble_context()` builds one Pydantic `AgentContext`:

- `conversation.transcript`
- `vision` (filled later by the vision tool)
- `audio.transcript` + `speech_features` + `emotion`
- `location` (lat/lng/accuracy from the browser)
- `session_id`, `user_id`, `timestamp`

Session keys (Redis or memory), TTL ~90s:

- `frames` — list of `{ timestamp_ms, offset_ms, data }` JPEG base64
- `location`
- `speech_features`
- `transcript`

Empty `frames: []` does not overwrite a previous good capture.

---

## 10. LangGraph brain

State (`AgentState`): `session_id`, `user_id`, `transcript`, `visual_context`, `audio_context`, `location_context`, `retrieved_memories`, `recognized_entities`, `saved_entity`, `intent`, `required_tools`, `response`.

```
START
  → normalize_context
  → detect_intent          # regex EN+HI; vision also if frames exist
  → gather_context         # parallel vision/memory/location, then entities
  → agent_reason           # one chat completion
  → response_prepare
END
```

`gather_context` runs vision / GPS / facts memory in parallel, then **sequentially** extracts an introduction (if any) and recalls named entities using transcript + scene + object labels. Entity save waits for vision so the stored description matches the current camera view.

### 10.1 Intent router (deterministic, no extra LLM)

| Signal | Need |
| --- | --- |
| look / see / phone / दिख / फोन / ये क्या / … **or any stored frames** | `vision` |
| where am i / कहाँ / लोकेशन | `location` |
| my name / नाम / याद | `memory` (facts) |
| this is / meet / ये हैं / नाम है / याद रख | `entity_save` (+ vision) |
| who is / कौन है / याद है / ये कौन **or frames** | `entity_recall` |

Vision is re-run every turn when frames exist, so a phone shown *after* “how are you” is not ignored. Introductions and “who is this?” also force vision.

`intent` is `casual`, `vision`, `location`, `memory`, or `mixed` when more than one need is present.

### 10.2 Tools (`app/agent/tools.py`)

| Tool | Permission | Behavior |
| --- | --- | --- |
| `analyze_current_scene` | private_read | Multi-frame vision JSON |
| `search_personal_memory` | private_read | Embedding search, `user_id` filter |
| `get_user_profile` | private_read | Profile + preference facts |
| `save_memory` | private_write | Embed + insert a free-text fact |
| `remember_entity` | private_write | Upsert named person / object / animal |
| `recall_entities` | private_read | Embedding + name match, `user_id` filter |
| `list_known_entities` | private_read | Recent named entities for this user |
| `get_current_location` | sensitive_read | Session GPS + Nominatim |
| `get_current_time` | read | UTC timestamp |

Realtime has **no** tools. LangGraph calls these from Python.

### 10.3 Vision

- Deduplicate frames (hash of JPEG prefix)
- Cap at 5 images
- One vision chat request with all images + the user question
- Output: `scene`, `objects[]`, `confidence`, `verified_from_frames`, `notes`
- The reasoner must use this if `scene` exists; it must not claim the camera is off

### 10.4 Facts memory

Local SQLite table `personal_memories`: `user_id`, `memory_type` (`profile` / `preference` / `episodic`), `content`, `importance`, `embedding` (JSON vector).

- Search: cosine similarity in Python, `user_id == current user` always
- Seed script facts: name Nitin Singh, prefers concise technical answers, likes Python, building Mavic / Video Agent Buddy, working on Bitconnect

Postgres + `pgvector` (`<=>`) is used when `DATABASE_URL` is not SQLite.

This store is for **facts about the user** (“I prefer short answers”). Named people, pets, and belongings use `known_entities` instead.

### 10.5 Named entities (people, objects, animals)

Module: `app/memory/entities.py`. Table `known_entities`:

| Column | Role |
| --- | --- |
| `id` | `ent_…` |
| `user_id` | Isolation key (never chosen by the LLM) |
| `kind` | `person` \| `object` \| `animal` |
| `name` | Display name (Rahul, Bruno, wallet) |
| `visual_description` | Last camera scene text attached at save time |
| `notes` | Original intro utterance |
| `embedding` | Vector of `kind named {name}. {visual}. {notes}` |
| `created_at` / `last_seen_at` | Upsert refreshes `last_seen_at` |

**Introduce** (examples): “this is Rahul”, “ये मेरे दोस्त राहुल हैं”, “this is my dog Bruno”, “this is my black wallet”, “meet Priya”, “its name is Milo”.

Flow:

1. Regex (EN + Hindi) extracts `name` + `kind`. If the utterance looks like an intro but regex misses the name, a small JSON chat completion extracts `{name, kind, notes}`.
2. `remember_entity` upserts on `(user_id, lower(name))`. A second intro of the same name appends new visual notes instead of duplicating the row.
3. `saved_entity` is passed into `agent_reason` so the spoken reply can confirm the name.

**Recall** (examples): “who is this?”, “कौन है?”, “do you remember Bruno?”, or any turn that already has camera frames.

Query = transcript + vision `scene` + object labels. Hits with cosine ≥ ~0.28 are returned. If the query contains the stored name, score is boosted (~0.92) so name recall works even when the camera scene has changed.

**Reasoner rules:**

- If `saved_entity` is set, confirm you now know that named person / object / animal.
- If `recognized_entities` has a strong match (score above ~0.45), identify them by that stored name.
- Never invent names that are not in memory or in the user’s current words.
- This is **not** biometric face ID. Recognition is “user named this earlier + similar description / name in the question”.

### 10.6 Location

Browser coordinates only. Reverse geocode via Nominatim (`User-Agent: MavicAgentBuddy/1.0`). The model is instructed never to invent lat/lng.

---

## 11. Language policy

Shared string `LANGUAGE_POLICY` (`app/agent/language.py`):

- User speaks Hindi and English (often Hinglish)
- Hindi in **Devanagari**, never Urdu/Arabic script
- Match the user’s language
- If a transcript arrives in Urdu script, treat it as Hindi and answer in Devanagari or English

Applied to both LangGraph and the Realtime speaker instructions. Vision and intent regexes include Hindi cues (दिख, फोन, कहाँ, नाम, कौन, याद, …).

---

## 12. Security (V1)

| Rule | Implementation |
| --- | --- |
| `user_id` not from the LLM | Injected by FastAPI `current_user_id` |
| Memory isolation | `WHERE user_id = …` on facts **and** entities |
| API key not in the browser | Ephemeral Realtime session via server SDP |
| OpenAI safety id | HMAC of `user_id` on `/v1/realtime/calls` |
| Frames not archived | Session TTL only; no S3 in V1 |
| Tool permission labels | Present for later authz; not enforced as a gateway yet |
| JWT | Ignored in development (JWT login is not implemented) |

Production still needs real auth, HTTPS, and a permission check before `save_memory` / `remember_entity`.

---

## 13. Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Chat, vision, embeddings, Realtime |
| `OPENAI_REALTIME_MODEL` | Default `gpt-realtime-2.1` |
| `OPENAI_CHAT_MODEL` / `OPENAI_VISION_MODEL` | Default `gpt-4.1-mini` |
| `OPENAI_EMBEDDING_MODEL` | Default `text-embedding-3-small` |
| `DATABASE_URL` | Default `sqlite:///./mavic.db` |
| `REDIS_URL` | Optional |
| `DEFAULT_USER_ID` / `DEFAULT_USER_NAME` | Demo identity |
| `APP_ENV` | `development` enables demo user |
| `CORS_ORIGINS` | Plus localhost / 127.0.0.1:3000 |
| `JWT_SECRET` | Unused in development |

---

## 14. Local run

```bash
# .env: OPENAI_API_KEY=sk-...
cd backend
source .venv/bin/activate
PYTHONPATH=. python3 scripts/seed_memory.py
uvicorn app.main:app --reload --port 8000

cd frontend
npm run dev   # http://localhost:3000
```

Allow camera, microphone, and location. Try:

- “How are you?” / “हेलो”
- “What is my name?” / “मेरा नाम क्या है?”
- “Where am I?”
- Hold an object: “ये कौन सा फोन है?” / “What am I holding?”
- Introduce, then recall: “This is Rahul” → later “Who is this?” / “कौन है?”
- “This is my dog Bruno” → “Do you remember Bruno?”
- “This is my black wallet” → show it again and ask what it is

Restart the backend once after pulling entity-memory changes so `known_entities` is created.

---

## 15. Failure modes (known)

| Symptom | Cause / fix |
| --- | --- |
| User bubble, no assistant | `/api/turn` 500 — check Uvicorn traceback |
| “I can’t see the phone” | Frames not uploaded before the turn, or Hindi not routed to vision (both addressed in current code) |
| Does not remember a person/object | Intro phrase not detected, empty name, or backend not restarted so `known_entities` is missing |
| Invents a name | Reasoner ignoring score threshold — only stored names / user words are allowed |
| Urdu script captions | Transcription language; reconnect after language policy |
| `turnLock is not defined` | Frontend ref missing (fixed) |
| Docker not found | SQLite path; Docker optional |
| Python 3.9 `str \| None` | Use `Optional[]` on Pydantic/SQLAlchemy models |

---

## 16. What V1 is not

- Not continuous video understanding
- Not biometric face / pet recognition (names come from the user)
- Not a proactive companion (“new person walked in”)
- Not multi-user production auth
- Not Kubernetes / Kafka / per-sense microservices

Those are V2/V3: separate perception workers, hot Redis context, Postgres facts, vector long-term memory at scale, human-in-the-loop for destructive tools.

---

## 17. Design rules (non-negotiable)

1. Perception ≠ reasoning.
2. Sample frames; do not stream video into LangGraph.
3. Verify across multiple frames.
4. Emotion is probabilistic context.
5. GPS is a tool result, not an LLM guess.
6. Facts memory and named entities are always namespaced by authenticated `user_id`.
7. Do not invent entity names; only recall what was stored or just said.
8. Realtime is the nervous system; LangGraph is the brain.
9. Optimize for **low perceived latency** (status: Looking / Thinking, then speech), not zero latency.
