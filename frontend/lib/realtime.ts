export type RealtimeEvent = {
  type: string;
  call_id?: string;
  name?: string;
  arguments?: string;
  transcript?: string;
  error?: { message?: string };
  delta?: string;
  item?: {
    type?: string;
    role?: string;
    call_id?: string;
    name?: string;
    arguments?: string;
    content?: Array<{ transcript?: string; type?: string; text?: string }>;
  };
};

export function eventTranscript(event: RealtimeEvent): string {
  return (
    (typeof event.transcript === "string" && event.transcript) ||
    (typeof event.delta === "string" && event.delta) ||
    event.item?.content?.[0]?.transcript ||
    event.item?.content?.[0]?.text ||
    ""
  );
}

export async function connectRealtime(opts: {
  sessionId: string;
  audioTrack: MediaStreamTrack;
  onEvent: (event: RealtimeEvent) => void;
  onRemoteStream: (stream: MediaStream) => void;
}) {
  const pc = new RTCPeerConnection();
  pc.addTransceiver("audio", { direction: "recvonly" });
  pc.ontrack = (event) => {
    if (event.streams[0]) opts.onRemoteStream(event.streams[0]);
  };
  pc.addTrack(opts.audioTrack);

  const dc = pc.createDataChannel("oai-events");
  const queued: string[] = [];

  function send(event: unknown) {
    const payload = JSON.stringify(event);
    if (dc.readyState === "open") {
      dc.send(payload);
      return;
    }
    queued.push(payload);
  }

  dc.addEventListener("open", () => {
    for (const payload of queued) dc.send(payload);
    queued.length = 0;
  });
  dc.addEventListener("message", (message) => {
    try {
      opts.onEvent(JSON.parse(message.data) as RealtimeEvent);
    } catch {
      /* ignore malformed */
    }
  });

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);

  let sdpResponse: Response | null = null;
  let lastError = "Voice handshake failed";
  for (let attempt = 0; attempt < 3; attempt += 1) {
    try {
      sdpResponse = await fetch("/api/realtime/sdp", {
        method: "POST",
        headers: { "Content-Type": "application/sdp" },
        body: offer.sdp ?? "",
      });
      if (sdpResponse.ok) break;
      lastError = (await sdpResponse.text()) || sdpResponse.statusText;
      if (sdpResponse.status < 500 && sdpResponse.status !== 429) {
        throw new Error(lastError);
      }
    } catch (err) {
      lastError = err instanceof Error ? err.message : lastError;
      if (attempt === 2) throw new Error(lastError);
    }
    await new Promise((resolve) => window.setTimeout(resolve, 800 * (attempt + 1)));
  }
  if (!sdpResponse?.ok) {
    throw new Error(lastError);
  }
  const answer = await sdpResponse.text();
  await pc.setRemoteDescription({ type: "answer", sdp: answer });

  await new Promise<void>((resolve, reject) => {
    if (dc.readyState === "open") {
      resolve();
      return;
    }
    const timer = window.setTimeout(() => reject(new Error("Voice data channel timed out")), 12000);
    dc.addEventListener(
      "open",
      () => {
        window.clearTimeout(timer);
        resolve();
      },
      { once: true }
    );
  });

  function speak(text: string) {
    if (!text.trim()) return;
    send({
      type: "response.create",
      response: {
        conversation: "none",
        output_modalities: ["audio"],
        instructions: `Read this reply aloud exactly, word for word. Do not add a greeting, commentary, or these instructions.\n\n${text}`,
      },
    });
  }

  return { pc, dc, send, speak };
}
