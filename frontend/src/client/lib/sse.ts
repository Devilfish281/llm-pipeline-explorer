/**
 * Client-side SSE (Server-Sent Events) reader — the browser half of the streaming pipeline.
 *
 * Every feature in this app streams results from the server via SSE. This module handles
 * the client side: sends a POST request, reads the streaming response chunk by chunk,
 * parses SSE events, and calls `onEvent` for each one.
 *
 * Three parsing modes:
 * - "json" — each `data:` line is a standalone JSON object
 * - "json-envelope" — each complete named event becomes `{ event, data }`
 * - "multiline" — events can span multiple `data:` lines as raw text
 *
 * Flow: `readSSE(options)` → POST to endpoint → stream chunks → parse events → `onEvent(parsed)`
 *
 * @see {@link file://src/server/lib/sse.ts} for the server-side emitter
 */
import { parseError } from "./parse-error.js";

export type SSEMode = "json" | "json-envelope" | "multiline";

export type SSEJSONEnvelope<TData = unknown> = {
  event: string;
  data: TData;
};

export type SSEOptions<TEvent = Record<string, unknown>> = {
  endpoint: string;
  body: unknown;
  onEvent: (parsed: TEvent) => void;
  onOpen?: () => void;
  mode?: SSEMode;
};

export type SSEResult = { ok: true } | { ok: false; error: string };

function readSSEFieldValue(
  line: string,
  fieldName: "event" | "data",
): string | null {
  const prefix = `${fieldName}:`;

  if (!line.startsWith(prefix)) {
    return null;
  }

  const value = line.slice(prefix.length);

  return value.startsWith(" ") ? value.slice(1) : value;
}

function parseJSONEnvelopeBlock<TEvent>(block: string): TEvent | null {
  let event = "";
  const dataLines: string[] = [];

  for (const line of block.split(/\r?\n/)) {
    const eventValue = readSSEFieldValue(line, "event");

    if (eventValue !== null) {
      event = eventValue;
      continue;
    }

    const dataValue = readSSEFieldValue(line, "data");

    if (dataValue !== null) {
      dataLines.push(dataValue);
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  const dataText = dataLines.join("\n");

  return {
    event,
    data: JSON.parse(dataText),
  } as TEvent;
}

function consumeJSONEnvelopeBlocks<TEvent>(
  buffer: string,
  onEvent: (parsed: TEvent) => void,
  flushRemainder: boolean,
): string {
  const eventSeparator = /\r?\n\r?\n/g;
  let blockStart = 0;
  let separatorMatch: RegExpExecArray | null;

  while ((separatorMatch = eventSeparator.exec(buffer)) !== null) {
    const block = buffer.slice(blockStart, separatorMatch.index);

    const parsed = parseJSONEnvelopeBlock<TEvent>(block);

    if (parsed !== null) {
      onEvent(parsed);
    }

    blockStart = eventSeparator.lastIndex;
  }

  const remainder = buffer.slice(blockStart);

  if (!flushRemainder) {
    return remainder;
  }

  if (remainder.length > 0) {
    const parsed = parseJSONEnvelopeBlock<TEvent>(remainder);

    if (parsed !== null) {
      onEvent(parsed);
    }
  }

  return "";
}

/**
 * Sends a POST request and reads the SSE response stream, invoking `onEvent`
 * for each parsed event.
 *
 * Returns `{ ok: true }` on success, or `{ ok: false, error: string }`
 * if the HTTP request fails.
 */
export async function readSSE<TEvent = Record<string, unknown>>(
  options: SSEOptions<TEvent>,
): Promise<SSEResult> {
  const { endpoint, body, onEvent, onOpen, mode = "json" } = options;

  const response = await fetch(endpoint, {
    body: JSON.stringify(body),
    headers: {
      "Content-Type": "application/json",
    },
    method: "POST",
  });

  if (!response.ok) {
    const error = await parseError(response);

    return {
      ok: false,
      error,
    };
  }

  onOpen?.();

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      break;
    }

    buffer += decoder.decode(value, {
      stream: true,
    });

    if (mode === "json-envelope") {
      buffer = consumeJSONEnvelopeBlocks(buffer, onEvent, false);

      continue;
    }

    if (mode === "multiline") {
      const messages = buffer.split("\n\n");
      buffer = messages.pop()!;

      for (const msg of messages) {
        const lines = msg.split("\n");
        let event = "";
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            event = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            dataLines.push(line.slice(6));
          } else if (line === "data:") {
            dataLines.push("");
          }
        }

        if (dataLines.length === 0) {
          continue;
        }

        const data = dataLines.join("\n");

        if (data === "") {
          continue;
        }

        onEvent({
          event,
          data,
        } as TEvent);
      }
    } else {
      const lines = buffer.split("\n");
      buffer = lines.pop()!;

      for (const line of lines) {
        if (line.startsWith("event:")) {
          continue;
        }

        if (line.startsWith("data: ")) {
          const parsed = JSON.parse(line.slice(6)) as TEvent;

          onEvent(parsed);
        }
      }
    }
  }

  if (mode === "json-envelope") {
    buffer += decoder.decode();

    consumeJSONEnvelopeBlocks(buffer, onEvent, true);
  }

  return {
    ok: true,
  };
}
