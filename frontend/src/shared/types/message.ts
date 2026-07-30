import type { Child } from "hono/jsx";

export type MessageRole = "user" | "assistant";

export type Message = {
  id: string;
  role: MessageRole;
  content: Child;
};
