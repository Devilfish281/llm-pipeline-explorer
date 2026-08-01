/**
 * Extracts a human-readable error message from a failed HTTP response.
 * Handles nested JSON error structures, safe string-valued FastAPI detail
 * responses, and falls back to the complete raw response text.
 */
export async function parseError(response: Response): Promise<string> {
  const text = await response.text();

  try {
    const json = JSON.parse(text);

    if (json.error?.message) {
      const parsed = JSON.parse(json.error.message);

      if (Array.isArray(parsed)) {
        return parsed
          .map((error: { message?: string }) => error.message)
          .filter(Boolean)
          .join(", ");
      }

      return json.error.message;
    }

    if (typeof json.detail === "string") {
      return json.detail;
    }

    return text;
  } catch {
    return text;
  }
}
