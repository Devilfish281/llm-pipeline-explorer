import { describe, expect, it } from "vitest";

import { parseError } from "./parse-error.js";

function createErrorResponse(body: string, status = 400): Response {
  return new Response(body, {
    status,
    headers: {
      "Content-Type": "application/json",
    },
  });
}

describe("parseError", () => {
  it("returns a safe string-valued FastAPI detail without raw JSON", async () => {
    const safeMessage = "Another Transformer request is already running.";

    const response = createErrorResponse(
      JSON.stringify({
        detail: safeMessage,
      }),
      429,
    );

    const result = await parseError(response);

    expect(result).toBe(safeMessage);
    expect(result).not.toContain('{"detail"');
    expect(result).not.toContain('"detail":');
  });

  it("preserves the existing nested error-message array behavior", async () => {
    const response = createErrorResponse(
      JSON.stringify({
        error: {
          message: JSON.stringify([
            {
              message: "First validation problem.",
            },
            {
              message: "Second validation problem.",
            },
            {
              code: "ignored-without-message",
            },
          ]),
        },
      }),
      422,
    );

    const result = await parseError(response);

    expect(result).toBe(
      "First validation problem., Second validation problem.",
    );
  });

  it("keeps non-string FastAPI detail responses as complete raw text", async () => {
    const body = JSON.stringify({
      detail: [
        {
          type: "missing",
          loc: ["body", "prompt"],
          msg: "Field required",
        },
      ],
    });

    const response = createErrorResponse(body, 422);

    const result = await parseError(response);

    expect(result).toBe(body);
  });

  it("falls back to the complete raw response when the body is not JSON", async () => {
    const body = "Internal Server Error";
    const response = createErrorResponse(body, 500);

    const result = await parseError(response);

    expect(result).toBe(body);
  });
});
