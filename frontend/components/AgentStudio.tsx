"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, streamTurn } from "@/lib/api";
import { createAudioAnalyzer, type SpeechFeatures } from "@/lib/audio";
import { createFrameSampler } from "@/lib/frames";
import { requestLocation, type GpsFix } from "@/lib/location";
import { connectRealtime, eventTranscript } from "@/lib/realtime";

type Status = "idle" | "connecting" | "listening" | "looking" | "thinking" | "speaking";
type ChatTurn = { role: "user" | "assistant" | "system"; text: string; streaming?: boolean };

function isJunkTranscript(text: string) {
  const trimmed = text.trim();
  if (trimmed.length < 2) return true;
  if (/^context:/i.test(trimmed)) return true;
  return /transcribe hindi|never use urdu|arabic script|keep english words|context:\s*#/i.test(trimmed);
}

function isInstructionLeak(text: string) {
  return /read this reply|speak this reply|do not add a greeting|do not call tools/i.test(text);
}

export function AgentStudio() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const remoteAudioRef = useRef<HTMLAudioElement>(null);
  const samplerRef = useRef<ReturnType<typeof createFrameSampler> | null>(null);
  const analyzerRef = useRef<ReturnType<typeof createAudioAnalyzer> | null>(null);
  const sessionIdRef = useRef<string>("");
  const locationRef = useRef<GpsFix | null>(null);
  const speakRef = useRef<(text: string) => void>(() => undefined);
  const turnLock = useRef(false);
  const queuedTranscript = useRef<string | null>(null);
  const capturePromise = useRef<Promise<void>>(Promise.resolve());
  const idleTimer = useRef<number | null>(null);

  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string>("");
  const [sessionId, setSessionId] = useState("");
  const [location, setLocation] = useState<GpsFix | null>(null);
  const [emotion, setEmotion] = useState("awaiting voice");
  const [features, setFeatures] = useState<SpeechFeatures>({ energy: 0, speech_rate: 0, pitch_variance: 0 });
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [textInput, setTextInput] = useState("");
  const [userCaption, setUserCaption] = useState("");
  const [assistantCaption, setAssistantCaption] = useState("");
  const spokenRef = useRef("");
  const replyRef = useRef("");
  const userSpeechRef = useRef("");
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  const pushTurn = useCallback((turn: ChatTurn) => {
    setTurns((current) => [...current.slice(-18), turn]);
  }, []);

  const upsertStreaming = useCallback((role: ChatTurn["role"], text: string, done = false) => {
    setTurns((current) => {
      const next = [...current];
      const last = next[next.length - 1];
      if (last && last.role === role && last.text === text && !last.streaming && done) {
        return current;
      }
      if (last && last.role === role && last.streaming) {
        next[next.length - 1] = { role, text, streaming: !done };
        return next.slice(-18);
      }
      return [...next.slice(-18), { role, text, streaming: !done }];
    });
  }, []);

  const syncPerception = useCallback(
    async (opts?: { frames?: boolean; transcript?: string }) => {
      const id = sessionIdRef.current;
      if (!id) return;
      const frames =
        opts?.frames && samplerRef.current
          ? await samplerRef.current.captureNow()
          : [];
      const snapshot = analyzerRef.current?.sample();
      if (snapshot) setFeatures(snapshot);
      await api("/api/context", {
        method: "POST",
        body: JSON.stringify({
          session_id: id,
          frames,
          location: locationRef.current,
          speech_features: snapshot,
          transcript: opts?.transcript,
        }),
      });
    },
    []
  );

  const runVoiceTurn = useCallback(
    async (transcript: string) => {
      if (!transcript.trim() || !sessionIdRef.current) return;
      if (turnLock.current) {
        queuedTranscript.current = transcript;
        return;
      }
      turnLock.current = true;
      setStatus("looking");
      setUserCaption("");
      try {
        await capturePromise.current.catch(() => undefined);
        await syncPerception({ frames: true, transcript });
        setStatus("thinking");
        let assembled = "";
        upsertStreaming("assistant", "", false);
        const result = await streamTurn(
          {
            session_id: sessionIdRef.current,
            transcript,
            location: locationRef.current,
            speech_features: analyzerRef.current?.sample(),
          },
          {
            onToken: (piece) => {
              assembled += piece;
              upsertStreaming("assistant", assembled, false);
              setAssistantCaption(assembled);
            },
          }
        );
        if (result.emotion?.phrasing) setEmotion(result.emotion.phrasing);
        if (result.junk) {
          setTurns((current) => current.filter((turn) => !(turn.role === "assistant" && !turn.text)));
          setAssistantCaption("");
          setStatus("listening");
          return;
        }
        const reply = (result.response || assembled).trim();
        upsertStreaming("assistant", reply, true);
        if (reply) {
          replyRef.current = reply;
          spokenRef.current = "";
          speakRef.current(reply);
          setStatus("speaking");
        } else {
          setAssistantCaption("");
          setStatus("listening");
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : "Agent turn failed";
        setError(message);
        pushTurn({ role: "system", text: message });
        setStatus("listening");
      } finally {
        turnLock.current = false;
        const next = queuedTranscript.current;
        queuedTranscript.current = null;
        if (next && next !== transcript) {
          void runVoiceTurn(next);
        }
      }
    },
    [pushTurn, syncPerception, upsertStreaming]
  );

  const start = useCallback(async () => {
    setError("");
    setStatus("connecting");
    try {
      const session = await api<{ session_id: string }>("/api/session", { method: "POST" });
      sessionIdRef.current = session.session_id;
      setSessionId(session.session_id);

      const media = await navigator.mediaDevices.getUserMedia({
        audio: true,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      if (videoRef.current) {
        videoRef.current.srcObject = media;
        await videoRef.current.play();
        samplerRef.current = createFrameSampler(videoRef.current);
      }
      analyzerRef.current = createAudioAnalyzer(media);

      try {
        const gps = await requestLocation();
        locationRef.current = gps;
        setLocation(gps);
      } catch {
        pushTurn({ role: "system", text: "Location permission denied — GPS tool will be unavailable." });
      }

      const audioTrack = media.getAudioTracks()[0];
      try {
        const realtime = await connectRealtime({
          sessionId: session.session_id,
          audioTrack,
          onRemoteStream: (stream) => {
            if (remoteAudioRef.current) {
              remoteAudioRef.current.srcObject = stream;
              remoteAudioRef.current.autoplay = true;
              void remoteAudioRef.current.play();
            }
          },
          onEvent: (event) => {
            if (event.type === "error") {
              setError(event.error?.message || "Realtime error");
            }
            if (event.type === "input_audio_buffer.speech_started") {
              setStatus("looking");
              userSpeechRef.current = "";
              setUserCaption("");
              capturePromise.current = syncPerception({ frames: true });
            }
            if (event.type === "input_audio_buffer.speech_stopped") {
              setStatus("thinking");
            }
            if (
              event.type === "response.audio.delta" ||
              event.type === "response.output_audio.delta" ||
              event.type === "output_audio_buffer.started"
            ) {
              setStatus("speaking");
            }
            if (event.type === "response.done") {
              setStatus("listening");
              setAssistantCaption("");
              spokenRef.current = "";
            }
            if (
              event.type === "response.output_audio_transcript.delta" ||
              event.type === "response.audio_transcript.delta" ||
              event.type === "response.output_text.delta"
            ) {
              const piece = eventTranscript(event);
              if (piece && !isInstructionLeak(piece)) {
                spokenRef.current += piece;
                setAssistantCaption(spokenRef.current);
                setStatus("speaking");
              }
            }
            if (
              event.type === "conversation.item.input_audio_transcription.completed" ||
              event.type === "conversation.item.input_audio_transcription.delta"
            ) {
              const completed = event.type.endsWith("completed");
              const full = event.transcript || event.item?.content?.[0]?.transcript || "";
              const piece = typeof event.delta === "string" ? event.delta : "";
              if (completed) {
                userSpeechRef.current = full || userSpeechRef.current;
              } else if (full && full.length >= userSpeechRef.current.length) {
                userSpeechRef.current = full;
              } else if (piece) {
                userSpeechRef.current += piece;
              }
              const text = userSpeechRef.current.trim();
              if (!text || isJunkTranscript(text)) {
                if (completed) userSpeechRef.current = "";
                return;
              }
              setUserCaption(text);
              upsertStreaming("user", text, completed);
              if (completed) {
                userSpeechRef.current = "";
                void runVoiceTurn(text);
              }
            }
          },
        });
        speakRef.current = realtime.speak;
      } catch (voiceErr) {
        setError(
          `${voiceErr instanceof Error ? voiceErr.message : "Realtime voice failed"} — camera + typed turns still work.`
        );
      }

      idleTimer.current = window.setInterval(() => {
        samplerRef.current?.grab(-1);
        const snapshot = analyzerRef.current?.sample();
        if (snapshot) {
          setFeatures(snapshot);
          const mood =
            snapshot.energy < 0.28
              ? "may sound tired"
              : snapshot.energy > 0.7
                ? "may sound intense"
                : "likely calm";
          setEmotion(mood);
        }
      }, 1400);

      setStatus("listening");
    } catch (err) {
      setStatus("idle");
      setError(err instanceof Error ? err.message : "Failed to start");
    }
  }, [pushTurn, runVoiceTurn, syncPerception, upsertStreaming]);

  async function sendText() {
    const transcript = textInput.trim();
    if (!transcript || !sessionIdRef.current) return;
    setTextInput("");
    setUserCaption("");
    upsertStreaming("user", transcript, true);
    setStatus("thinking");
    await syncPerception({ frames: true, transcript });
    let assembled = "";
    upsertStreaming("assistant", "", false);
    try {
      const result = await streamTurn(
        {
          session_id: sessionIdRef.current,
          transcript,
          location: locationRef.current,
          speech_features: features,
        },
        {
          onToken: (piece) => {
            assembled += piece;
            upsertStreaming("assistant", assembled, false);
            setAssistantCaption(assembled);
          },
        }
      );
      if (result.emotion?.phrasing) setEmotion(result.emotion.phrasing);
      const reply = (result.response || assembled).trim();
      upsertStreaming("assistant", reply, true);
      if (reply) {
        replyRef.current = reply;
        spokenRef.current = "";
        speakRef.current(reply);
        setStatus("speaking");
      } else {
        setAssistantCaption("");
        setStatus(sessionId ? "listening" : "idle");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Agent turn failed";
      setError(message);
      setStatus(sessionId ? "listening" : "idle");
    }
  }

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, userCaption, assistantCaption]);

  useEffect(() => {
    return () => {
      analyzerRef.current?.close();
      if (idleTimer.current) window.clearInterval(idleTimer.current);
    };
  }, []);

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">Mavic Agent</p>
          <h1>Video Agent Buddy</h1>
        </div>
        <div className="chips">
          <span className={`chip ${status}`}>{statusLabel(status)}</span>
          <span className="chip">{emotion}</span>
          <span className="chip">
            {location
              ? `${location.latitude.toFixed(3)}, ${location.longitude.toFixed(3)} ±${Math.round(location.accuracy_m)}m`
              : "GPS off"}
          </span>
        </div>
      </header>

      <section className="stage">
        <div className="eye">
          <video ref={videoRef} playsInline muted className="camera" />
          {status === "idle" && (
            <div className="idle-cover">
              <p>Camera is the eyes. Microphone is the ears. Allow access to start talking with Mavic.</p>
              <button className="wake" onClick={() => void start()}>
                Enable camera, mic, location
              </button>
            </div>
          )}
          <div className="hud">
            <span>EYES</span>
            <span>MULTI-FRAME</span>
            <span>{sessionId ? sessionId.slice(-8) : "NO SESSION"}</span>
          </div>
          <div className={`orb ${status}`} />
          {status !== "idle" && (userCaption || assistantCaption) && (
            <div className="captions">
              {userCaption && (
                <div className="live-line user">
                  <span className="who">YOU</span>
                  {userCaption}
                </div>
              )}
              {assistantCaption && (
                <div className="live-line assistant">
                  <span className="who">MAVIC</span>
                  {assistantCaption}
                </div>
              )}
            </div>
          )}
        </div>

        <aside className="panel">
          <div className="transcript">
            {turns.length === 0 && status !== "idle" && (
              <p className="placeholder">Listening. Ask anything — name, location, or what you are holding.</p>
            )}
            {turns.map((turn, i) => (
              <div key={`${turn.role}-${i}`} className={`bubble ${turn.role}${turn.streaming ? " streaming" : ""}`}>
                <small>{turn.role}</small>
                <p>{turn.text}</p>
              </div>
            ))}
            <div ref={transcriptEndRef} />
          </div>

          {status !== "idle" && (
            <div className="composer">
              <input
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") void sendText();
                }}
                placeholder="Type if you want a silent turn…"
              />
              <button onClick={() => void sendText()}>Send</button>
            </div>
          )}
          {error && <p className="error">{error}</p>}
          {status !== "idle" && (
            <p className="meters">
              energy {features.energy.toFixed(2)} · rate {features.speech_rate.toFixed(2)} · pitch{" "}
              {features.pitch_variance.toFixed(2)}
            </p>
          )}
        </aside>
      </section>
      <audio ref={remoteAudioRef} autoPlay playsInline />
    </main>
  );
}

function statusLabel(status: Status) {
  switch (status) {
    case "looking":
      return "Looking";
    case "thinking":
      return "Thinking";
    case "speaking":
      return "Speaking";
    case "connecting":
      return "Connecting";
    case "listening":
      return "Listening";
    default:
      return "Standby";
  }
}
