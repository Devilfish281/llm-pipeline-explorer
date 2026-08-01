import type { SSEJSONEnvelope, SSEMode, SSEResult } from "./sse.js";

import { afterEach, describe, expect, it, vi } from "vitest";

import { readSSE } from "./sse.js";

function createStreamResponse(chunks: string[]): Response {
  const encoder = new TextEncoder();

  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }

      controller.close();
    },
  });

  return new Response(body, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
    },
  });
}

async function collectSSEEvents<TEvent>(
  chunks: string[],
  mode: SSEMode,
): Promise<{
  events: TEvent[];
  result: SSEResult;
}> {
  const fetchMock = vi.fn(async () => createStreamResponse(chunks));

  vi.stubGlobal("fetch", fetchMock);

  const events: TEvent[] = [];

  const result = await readSSE<TEvent>({
    endpoint: "/api/test-stream",
    body: {
      request: "test",
    },
    mode,
    onEvent: (event) => events.push(event),
  });

  return {
    events,
    result,
  };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("readSSE json-envelope mode", () => {
  it("preserves a loaded event name and parses its joined JSON data", async () => {
    const { events, result } = await collectSSEEvents<
      SSEJSONEnvelope<{
        file: string;
        prompt: string;
      }>
    >(
      [
        "event: loaded\n" +
          'data: {"file":"model.json",\n' +
          'data: "prompt":"once upon a time"}\n\n',
      ],
      "json-envelope",
    );

    expect(result).toStrictEqual({
      ok: true,
    });

    expect(events).toStrictEqual([
      {
        event: "loaded",
        data: {
          file: "model.json",
          prompt: "once upon a time",
        },
      },
    ]);
  });

  it("preserves result, done, and error event names", async () => {
    const { events, result } = await collectSSEEvents<SSEJSONEnvelope<unknown>>(
      [
        "event: result\n" +
          'data: {"text":"once upon a time went home"}\n\n' +
          "event: done\n" +
          "data: {}\n\n" +
          "event: error\n" +
          'data: {"error":"The saved model could not be loaded."}\n\n',
      ],
      "json-envelope",
    );

    expect(result).toStrictEqual({
      ok: true,
    });

    expect(events).toStrictEqual([
      {
        event: "result",
        data: {
          text: "once upon a time went home",
        },
      },
      {
        event: "done",
        data: {},
      },
      {
        event: "error",
        data: {
          error: "The saved model could not be loaded.",
        },
      },
    ]);
  });

  it("emits multiple events from one chunk in order", async () => {
    const { events } = await collectSSEEvents<SSEJSONEnvelope<unknown>>(
      [
        "event: loaded\n" +
          'data: {"file":"model.json","prompt":"prompt"}\n\n' +
          "event: result\n" +
          'data: {"text":"prompt continuation"}\n\n' +
          "event: done\n" +
          "data: {}\n\n",
      ],
      "json-envelope",
    );

    expect(events.map((event) => event.event)).toStrictEqual([
      "loaded",
      "result",
      "done",
    ]);
  });

  it("reconstructs one event split across network chunks", async () => {
    const { events } = await collectSSEEvents<
      SSEJSONEnvelope<{
        file: string;
        prompt: string;
      }>
    >(
      [
        "event: loa",
        'ded\ndata: {"file":"model',
        '.json","prompt":"once upon',
        ' a time"}\n',
        "\n",
      ],
      "json-envelope",
    );

    expect(events).toStrictEqual([
      {
        event: "loaded",
        data: {
          file: "model.json",
          prompt: "once upon a time",
        },
      },
    ]);
  });

  it("accepts CRLF and LF event separators", async () => {
    const { events } = await collectSSEEvents<SSEJSONEnvelope<unknown>>(
      [
        "event: loaded\r\n" +
          'data: {"file":"model.json","prompt":"prompt"}\r\n' +
          "\r\n" +
          "event: done\n" +
          "data: {}\n\n",
      ],
      "json-envelope",
    );

    expect(events).toStrictEqual([
      {
        event: "loaded",
        data: {
          file: "model.json",
          prompt: "prompt",
        },
      },
      {
        event: "done",
        data: {},
      },
    ]);
  });

  it("flushes a final complete event without a trailing blank line", async () => {
    const { events } = await collectSSEEvents<
      SSEJSONEnvelope<Record<string, never>>
    >(["event: done\n", "data: {}"], "json-envelope");

    expect(events).toStrictEqual([
      {
        event: "done",
        data: {},
      },
    ]);
  });
});

describe("readSSE existing mode compatibility", () => {
  it("keeps json mode payload-only and ignores event names", async () => {
    const { events, result } = await collectSSEEvents<Record<string, unknown>>(
      [
        "event: loaded\n" +
          'data: {"file":"model.json","prompt":"prompt"}\n\n' +
          "event: done\n" +
          "data: {}\n\n",
      ],
      "json",
    );

    expect(result).toStrictEqual({
      ok: true,
    });

    expect(events).toStrictEqual([
      {
        file: "model.json",
        prompt: "prompt",
      },
      {},
    ]);
  });

  it("keeps multiline mode event and raw joined data behavior", async () => {
    const { events, result } = await collectSSEEvents<{
      event: string;
      data: string;
    }>(["event: word\n" + "data: hello\n" + "data: world\n\n"], "multiline");

    expect(result).toStrictEqual({
      ok: true,
    });

    expect(events).toStrictEqual([
      {
        event: "word",
        data: "hello\nworld",
      },
    ]);
  });
});
