export const TRAIN_TRANSFORMER_ENDPOINT = "/api/train-transformer" as const;
export const LOAD_TRANSFORMER_ENDPOINT = "/api/load-transformer" as const;

const FILE_COMMAND_PREFIX = "file:";

const STRICT_DECIMAL_PATTERN =
  /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/;

export const TRANSFORMER_FILE_COMMAND_USAGE =
  "Usage: File:<model file>|<starting prompt>|" +
  "<temperature> <top-p> <max tokens>. " +
  "Leave <model file> empty to use the newest valid saved model.";

export const TRANSFORMER_FILE_VALIDATION_MESSAGES = {
  sections:
    'A saved-model command must contain exactly three "|" separated sections. ' +
    TRANSFORMER_FILE_COMMAND_USAGE,
  prompt:
    "The starting prompt must not be empty. " + TRANSFORMER_FILE_COMMAND_USAGE,
  settingsCount:
    "Generation settings must contain exactly three values: " +
    "temperature, top-p, and maximum tokens. " +
    TRANSFORMER_FILE_COMMAND_USAGE,
  numeric:
    "Temperature, top-p, and maximum tokens must be valid finite decimal numbers. " +
    TRANSFORMER_FILE_COMMAND_USAGE,
  temperatureRange:
    "Temperature must be between 0.1 and 2.0. " +
    TRANSFORMER_FILE_COMMAND_USAGE,
  topPRange:
    "Top-p must be between 0.1 and 1.0. " + TRANSFORMER_FILE_COMMAND_USAGE,
  maxTokensInteger:
    "Maximum tokens must be an integer. " + TRANSFORMER_FILE_COMMAND_USAGE,
  maxTokensRange:
    "Maximum tokens must be between 3 and 500. " +
    TRANSFORMER_FILE_COMMAND_USAGE,
} as const;

type TransformerFileValidationMessage =
  (typeof TRANSFORMER_FILE_VALIDATION_MESSAGES)[keyof typeof TRANSFORMER_FILE_VALIDATION_MESSAGES];

export interface TransformerTrainingRequestBody {
  epochs: number;
  temperature: number;
  topP: number;
  numLayers: number;
  maxTokens: number;
}

export interface TransformerLoadRequestBody {
  modelFile: string | null;
  prompt: string;
  temperature: number;
  topP: number;
  maxTokens: number;
}

export interface TransformerTrainingSubmissionPlan {
  kind: "request";
  mode: "training";
  endpoint: typeof TRAIN_TRANSFORMER_ENDPOINT;
  body: TransformerTrainingRequestBody;
}

export interface TransformerLoadSubmissionPlan {
  kind: "request";
  mode: "load";
  endpoint: typeof LOAD_TRANSFORMER_ENDPOINT;
  body: TransformerLoadRequestBody;
}

export interface TransformerValidationSubmissionPlan {
  kind: "validation";
  assistantContent: string;
}

export type TransformerRequestSubmissionPlan =
  | TransformerTrainingSubmissionPlan
  | TransformerLoadSubmissionPlan;

export type TransformerSubmissionPlan =
  | TransformerRequestSubmissionPlan
  | TransformerValidationSubmissionPlan;

export function buildTransformerTrainingRequestBody(
  input: string,
): TransformerTrainingRequestBody {
  const values = input.trim().split(/\s+/);

  return {
    epochs: Number.parseInt(values[0] ?? "", 10) || 300,
    temperature: Number.parseFloat(values[1] ?? "") || 0.8,
    topP: Number.parseFloat(values[2] ?? "") || 0.9,
    numLayers: Number.parseInt(values[3] ?? "", 10) || 2,
    maxTokens: Number.parseInt(values[4] ?? "", 10) || 40,
  };
}

function createValidationSubmission(
  assistantContent: TransformerFileValidationMessage,
): TransformerValidationSubmissionPlan {
  return {
    kind: "validation",
    assistantContent,
  };
}

function parseStrictDecimal(token: string): number | null {
  if (!STRICT_DECIMAL_PATTERN.test(token)) {
    return null;
  }

  const value = Number(token);

  return Number.isFinite(value) ? value : null;
}

function planSavedTransformerSubmission(
  commandAfterPrefix: string,
): TransformerSubmissionPlan {
  const sections = commandAfterPrefix.split("|");

  if (sections.length !== 3) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.sections,
    );
  }

  const modelFileSection = sections[0] ?? "";
  const promptSection = sections[1] ?? "";
  const settingsSection = sections[2] ?? "";

  const prompt = promptSection.trim();

  if (prompt.length === 0) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.prompt,
    );
  }

  const trimmedSettings = settingsSection.trim();
  const settingTokens =
    trimmedSettings.length === 0 ? [] : trimmedSettings.split(/\s+/);

  if (settingTokens.length !== 3) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.settingsCount,
    );
  }

  const temperature = parseStrictDecimal(settingTokens[0] ?? "");
  const topP = parseStrictDecimal(settingTokens[1] ?? "");
  const maxTokens = parseStrictDecimal(settingTokens[2] ?? "");

  if (temperature === null || topP === null || maxTokens === null) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.numeric,
    );
  }

  if (temperature < 0.1 || temperature > 2.0) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.temperatureRange,
    );
  }

  if (topP < 0.1 || topP > 1.0) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.topPRange,
    );
  }

  if (!Number.isInteger(maxTokens)) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.maxTokensInteger,
    );
  }

  if (maxTokens < 3 || maxTokens > 500) {
    return createValidationSubmission(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.maxTokensRange,
    );
  }

  return {
    kind: "request",
    mode: "load",
    endpoint: LOAD_TRANSFORMER_ENDPOINT,
    body: {
      modelFile: modelFileSection === "" ? null : modelFileSection,
      prompt,
      temperature,
      topP,
      maxTokens,
    },
  };
}

export function planTransformerSubmission(
  input: string,
): TransformerSubmissionPlan {
  const commandWithoutLeadingWhitespace = input.trimStart();
  const possiblePrefix = commandWithoutLeadingWhitespace.slice(
    0,
    FILE_COMMAND_PREFIX.length,
  );

  if (possiblePrefix.toLowerCase() === FILE_COMMAND_PREFIX) {
    return planSavedTransformerSubmission(
      commandWithoutLeadingWhitespace.slice(FILE_COMMAND_PREFIX.length),
    );
  }

  return {
    kind: "request",
    mode: "training",
    endpoint: TRAIN_TRANSFORMER_ENDPOINT,
    body: buildTransformerTrainingRequestBody(input),
  };
}
