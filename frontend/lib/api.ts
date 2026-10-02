export const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  let response: Response;
  try {
    response = await fetch(url, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new Error("Cannot reach the agent backend. Is uvicorn running on port 8000?");
  }
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json() as Promise<T>;
  }
  return response.text() as Promise<T>;
}

export type TurnStreamDone = {
  response: string;
  emotion?: { phrasing?: string };
  junk?: boolean;
};

export async function streamTurn(
  body: unknown,
  handlers: {
    onMeta?: (meta: Record<string, unknown>) => void;
    onToken?: (text: string) => void;
    onDone?: (done: TurnStreamDone) => void;
  }
): Promise<TurnStreamDone> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/turn/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(body),
    });
  } catch {
    throw new Error("Cannot reach the agent backend. Is uvicorn running on port 8000?");
  }
  if (!response.ok || !response.body) {
    throw new Error((await response.text()) || response.statusText);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let donePayload: TurnStreamDone = { response: "" };

  const consume = (block: string) => {
    let eventName = "message";
    const dataLines: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
    }
    if (!dataLines.length) return;
    const data = JSON.parse(dataLines.join("\n"));
    if (eventName === "meta") handlers.onMeta?.(data);
    if (eventName === "token") handlers.onToken?.(data.text || "");
    if (eventName === "done") {
      donePayload = data as TurnStreamDone;
      handlers.onDone?.(donePayload);
    }
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";
    for (const block of parts) {
      if (block.trim()) consume(block);
    }
  }
  if (buffer.trim()) consume(buffer);
  return donePayload;
}

