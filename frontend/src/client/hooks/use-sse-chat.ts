/**
 * Generic hook for SSE-based chat features — the foundation for most pages in the app.
 *
 * Every feature (tokenize, embed, neural net, attention, etc.) follows the same pattern:
 * 1. User types a message and hits send
 * 2. A POST request streams SSE events from the server
 * 3. Each event updates the state and re-renders the result component
 *
 * This hook encapsulates that pattern. Each feature provides either:
 * - a static `endpoint` with an optional `buildBody(input)` callback; or
 * - `prepareSubmission(input)`, which returns a request or local validation result.
 *
 * Each feature also provides:
 * - `initState()` — creates fresh state for a new network request
 * - `onEvent(parsed, state)` — handles each SSE event, updates state, and returns content
 *
 * The hook manages message history, loading state, input, local validation,
 * and the streaming lifecycle.
 *
 * @see {@link file://src/client/lib/sse.ts} for the SSE reader this hook uses
 */
import type { Child } from "hono/jsx";
import type { Message } from "../../shared/types/message.js";
import type { SSEMode } from "../lib/sse.js";

import { useState } from "hono/jsx";
import { readSSE } from "../lib/sse.js";

export type SSERequestSubmission = {
  kind: "request";
  endpoint: string;
  body: unknown;
};

export type SSEValidationSubmission = {
  kind: "validation";
  assistantContent: Child;
};

export type SSEPreparedSubmission =
  | SSERequestSubmission
  | SSEValidationSubmission;

export type SSEMessageStart = (
  previous: Message[],
  userMessage: Message,
  assistantMessage: Message,
) => Message[];

type UseSSEChatCommonOptions<TState, TEvent> = {
  title: string;
  tagline: string;
  initState: () => TState;
  onEvent: (parsed: TEvent, state: TState) => Child | undefined;
  mode?: SSEMode;
  startMessages?: SSEMessageStart;
};

type UseSSEChatStaticRequestOptions = {
  endpoint: string;
  buildBody?: (input: string) => unknown;
  prepareSubmission?: never;
};

type UseSSEChatPreparedRequestOptions = {
  endpoint?: never;
  buildBody?: never;
  prepareSubmission: (input: string) => SSEPreparedSubmission;
};

export type UseSSEChatOptions<
  TState,
  TEvent = Record<string, unknown>,
> = UseSSEChatCommonOptions<TState, TEvent> &
  (UseSSEChatStaticRequestOptions | UseSSEChatPreparedRequestOptions);

export type UseSSEChatReturn = {
  input: string;
  loading: boolean;
  messages: Message[];
  sendMessage: () => Promise<void>;
  setInput: (value: string) => void;
  tagline: string;
  title: string;
};

export function useSSEChat<TState, TEvent = Record<string, unknown>>(
  options: UseSSEChatOptions<TState, TEvent>,
): UseSSEChatReturn {
  const {
    title,
    tagline,
    initState,
    onEvent,
    mode,
    startMessages = (previous, userMessage, assistantMessage) => [
      ...previous,
      userMessage,
      assistantMessage,
    ],
  } = options;

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const submission = options.prepareSubmission
      ? options.prepareSubmission(input)
      : {
          kind: "request" as const,
          endpoint: options.endpoint,
          body: (
            options.buildBody ?? ((value: string) => ({ message: value }))
          )(input),
        };

    const userMessage: Message = {
      content: input,
      id: crypto.randomUUID(),
      role: "user",
    };

    const assistantId = crypto.randomUUID();

    const assistantMessage: Message = {
      content:
        submission.kind === "validation" ? submission.assistantContent : "",
      id: assistantId,
      role: "assistant",
    };

    setMessages((previous) =>
      startMessages(previous, userMessage, assistantMessage),
    );

    setInput("");

    if (submission.kind === "validation") {
      setLoading(false);
      return;
    }

    setLoading(true);

    try {
      const state = initState();

      const result = await readSSE<TEvent>({
        endpoint: submission.endpoint,
        body: submission.body,
        mode,
        onOpen: () => setLoading(false),
        onEvent: (parsed) => {
          const content = onEvent(parsed, state);

          if (content !== undefined) {
            setMessages((previous) =>
              previous.map((message) =>
                message.id === assistantId ? { ...message, content } : message,
              ),
            );
          }
        },
      });

      if (!result.ok) {
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: `Error: ${result.error}`,
                }
              : message,
          ),
        );

        setLoading(false);
      }
    } catch (error) {
      console.error("SSE request failed:", error);

      const message = error instanceof Error ? error.message : String(error);

      setMessages((previous) => [
        ...previous,
        {
          content: `Something went wrong: ${message}`,
          id: crypto.randomUUID(),
          role: "assistant",
        },
      ]);

      setLoading(false);
    }
  };

  return {
    input,
    loading,
    messages,
    sendMessage,
    setInput,
    tagline,
    title,
  };
}
