export type BufferedFrame = {
  timestamp_ms: number;
  offset_ms: number;
  data: string;
};

const TARGET_WIDTH = 640;

export function createFrameSampler(video: HTMLVideoElement) {
  const canvas = document.createElement("canvas");
  const ring: BufferedFrame[] = [];

  function grab(offsetMs = 0): BufferedFrame | null {
    if (!video.videoWidth) return null;
    const scale = TARGET_WIDTH / video.videoWidth;
    canvas.width = TARGET_WIDTH;
    canvas.height = Math.round(video.videoHeight * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.72);
    const data = dataUrl.split(",")[1] ?? "";
    const frame: BufferedFrame = {
      timestamp_ms: Date.now(),
      offset_ms: offsetMs,
      data,
    };
    ring.push(frame);
    if (ring.length > 10) ring.shift();
    return frame;
  }

  function latest(n = 3): BufferedFrame[] {
    return ring.slice(-n);
  }

  async function captureUtteranceWindow(): Promise<BufferedFrame[]> {
    const pre = latest(2);
    const t0 = grab(0);
    await wait(200);
    const t200 = grab(200);
    await wait(300);
    const t500 = grab(500);
    const unique = new Map<string, BufferedFrame>();
    for (const frame of [...pre, t0, t200, t500]) {
      if (!frame) continue;
      unique.set(frame.data.slice(0, 80), frame);
    }
    return [...unique.values()].slice(-5);
  }

  async function captureNow(): Promise<BufferedFrame[]> {
    const grabbed: BufferedFrame[] = [];
    const first = grab(0);
    if (first) grabbed.push(first);
    await wait(140);
    const second = grab(140);
    if (second) grabbed.push(second);
    await wait(180);
    const third = grab(320);
    if (third) grabbed.push(third);
    const unique = new Map<string, BufferedFrame>();
    for (const frame of [...latest(2), ...grabbed]) {
      unique.set(frame.data.slice(32, 96), frame);
    }
    return [...unique.values()].slice(-5);
  }

  return { grab, latest, captureUtteranceWindow, captureNow, ring };
}

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
