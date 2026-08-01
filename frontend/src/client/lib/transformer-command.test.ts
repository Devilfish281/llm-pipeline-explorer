import type { Message } from "../../shared/types/message.js";
import type {
  TransformerLoadSubmissionPlan,
  TransformerTrainingSubmissionPlan,
  TransformerValidationSubmissionPlan,
} from "./transformer-command.js";

import { describe, expect, it } from "vitest";
import {
  LOAD_TRANSFORMER_ENDPOINT,
  TRAIN_TRANSFORMER_ENDPOINT,
  TRANSFORMER_FILE_COMMAND_USAGE,
  TRANSFORMER_FILE_VALIDATION_MESSAGES,
  planTransformerSubmission,
  replaceTransformerMessages,
} from "./transformer-command.js";

function requireTrainingPlan(input: string): TransformerTrainingSubmissionPlan {
  const plan = planTransformerSubmission(input);

  if (plan.kind !== "request" || plan.mode !== "training") {
    throw new Error(
      `Expected a training request for ${JSON.stringify(input)}, ` +
        `but received ${JSON.stringify(plan)}.`,
    );
  }

  return plan;
}

function requireLoadPlan(input: string): TransformerLoadSubmissionPlan {
  const plan = planTransformerSubmission(input);

  if (plan.kind !== "request" || plan.mode !== "load") {
    throw new Error(
      `Expected a load request for ${JSON.stringify(input)}, ` +
        `but received ${JSON.stringify(plan)}.`,
    );
  }

  return plan;
}

function requireValidationPlan(
  input: string,
): TransformerValidationSubmissionPlan {
  const plan = planTransformerSubmission(input);

  if (plan.kind !== "validation") {
    throw new Error(
      `Expected local validation for ${JSON.stringify(input)}, ` +
        `but received ${JSON.stringify(plan)}.`,
    );
  }

  return plan;
}

describe("numeric Transformer training commands", () => {
  it("preserves the exact full five-field training request", () => {
    expect(requireTrainingPlan("50 1.0 0.6 1 3")).toStrictEqual({
      kind: "request",
      mode: "training",
      endpoint: TRAIN_TRANSFORMER_ENDPOINT,
      body: {
        epochs: 50,
        temperature: 1,
        topP: 0.6,
        numLayers: 1,
        maxTokens: 3,
      },
    });
  });

  it.each([
    [
      "uses defaults for omitted trailing positions",
      "50 1.0",
      {
        epochs: 50,
        temperature: 1,
        topP: 0.9,
        numLayers: 2,
        maxTokens: 40,
      },
    ],
    [
      "uses defaults for tokens without numeric prefixes",
      "bad bad bad bad bad",
      {
        epochs: 300,
        temperature: 0.8,
        topP: 0.9,
        numLayers: 2,
        maxTokens: 40,
      },
    ],
    [
      "preserves permissive numeric-prefix conversion",
      "50abc 1.0junk 0.6x 1layer 3tokens",
      {
        epochs: 50,
        temperature: 1,
        topP: 0.6,
        numLayers: 1,
        maxTokens: 3,
      },
    ],
    [
      "preserves the current zero-like fallback behavior",
      "0 0 0 0 0",
      {
        epochs: 300,
        temperature: 0.8,
        topP: 0.9,
        numLayers: 2,
        maxTokens: 40,
      },
    ],
    [
      "ignores sixth and later positions",
      "50 1.0 0.6 1 3 ignored extra",
      {
        epochs: 50,
        temperature: 1,
        topP: 0.6,
        numLayers: 1,
        maxTokens: 3,
      },
    ],
    [
      "keeps ordinary non-File text on the training path",
      "hello",
      {
        epochs: 300,
        temperature: 0.8,
        topP: 0.9,
        numLayers: 2,
        maxTokens: 40,
      },
    ],
  ])("%s", (_description, input, expectedBody) => {
    const plan = requireTrainingPlan(input);

    expect(plan.endpoint).toBe("/api/train-transformer");
    expect(plan.body).toStrictEqual(expectedBody);
    expect(Object.keys(plan.body)).toStrictEqual([
      "epochs",
      "temperature",
      "topP",
      "numLayers",
      "maxTokens",
    ]);
  });
});

describe("saved Transformer command classification", () => {
  it.each([
    "File:model.json|once upon a time|0.8 0.9 3",
    "file:model.json|once upon a time|0.8 0.9 3",
    "FILE:model.json|once upon a time|0.8 0.9 3",
    "fIlE:model.json|once upon a time|0.8 0.9 3",
    "   File:model.json|once upon a time|0.8 0.9 3",
    "\tFile:model.json|once upon a time|0.8 0.9 3",
    "\nFile:model.json|once upon a time|0.8 0.9 3",
    " \t\n  FiLe:model.json|once upon a time|0.8 0.9 3",
  ])(
    "classifies a case-insensitive File prefix before numeric parsing: %s",
    (input) => {
      const plan = requireLoadPlan(input);

      expect(plan.endpoint).toBe(LOAD_TRANSFORMER_ENDPOINT);
      expect(plan.endpoint).toBe("/api/load-transformer");
    },
  );

  it.each([
    "Files:model.json|once upon a time|0.8 0.9 3",
    "File =model.json|once upon a time|0.8 0.9 3",
    "File :model.json|once upon a time|0.8 0.9 3",
    "prefix File:model.json|once upon a time|0.8 0.9 3",
    "aFile:model.json|once upon a time|0.8 0.9 3",
  ])("keeps a near-match on the numeric training path: %s", (input) => {
    const plan = requireTrainingPlan(input);

    expect(plan.endpoint).toBe(TRAIN_TRANSFORMER_ENDPOINT);
  });
});

describe("saved Transformer load requests", () => {
  it("constructs the exact named-model endpoint and five-field body", () => {
    const plan = requireLoadPlan(
      "File:transformer-weights-e100-l1-d32-h2-ff128-ctx32.json|" +
        "once upon a time|0.8 0.9 3",
    );

    expect(plan).toStrictEqual({
      kind: "request",
      mode: "load",
      endpoint: "/api/load-transformer",
      body: {
        modelFile: "transformer-weights-e100-l1-d32-h2-ff128-ctx32.json",
        prompt: "once upon a time",
        temperature: 0.8,
        topP: 0.9,
        maxTokens: 3,
      },
    });

    expect(Object.keys(plan.body)).toStrictEqual([
      "modelFile",
      "prompt",
      "temperature",
      "topP",
      "maxTokens",
    ]);

    expect(plan.body).not.toHaveProperty("useLatest");
    expect(plan.body).not.toHaveProperty("epochs");
    expect(plan.body).not.toHaveProperty("numLayers");
  });

  it("maps an exactly empty selector section to null", () => {
    const plan = requireLoadPlan("File:|once upon a time|0.8 0.9 3");

    expect(plan.body).toStrictEqual({
      modelFile: null,
      prompt: "once upon a time",
      temperature: 0.8,
      topP: 0.9,
      maxTokens: 3,
    });
  });

  it("preserves every character of a nonempty selector", () => {
    const selector = " Model Name.JSON \t";

    const plan = requireLoadPlan(`File:${selector}|once upon a time|0.8 0.9 3`);

    expect(plan.body.modelFile).toBe(selector);
  });

  it("trims only outer prompt whitespace and preserves the interior", () => {
    const plan = requireLoadPlan(
      "File:model.json|\t  Once   upon\ta time  \n|0.8 0.9 3",
    );

    expect(plan.body.prompt).toBe("Once   upon\ta time");
  });
});

describe("saved Transformer grammar validation", () => {
  it.each([
    [
      "rejects a command with no separators",
      "File:model.json",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.sections,
    ],
    [
      "rejects a command with only one separator",
      "File:model.json|once upon a time",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.sections,
    ],
    [
      "rejects an additional fourth section",
      "File:model.json|once upon a time|0.8 0.9 3|extra",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.sections,
    ],
    [
      "rejects a pipe inside the prompt",
      "File:model.json|once|upon a time|0.8 0.9 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.sections,
    ],
    [
      "rejects an empty prompt",
      "File:model.json||0.8 0.9 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.prompt,
    ],
    [
      "rejects a whitespace-only prompt",
      "File:model.json| \t\n |0.8 0.9 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.prompt,
    ],
    [
      "rejects missing generation settings",
      "File:model.json|once upon a time|",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.settingsCount,
    ],
    [
      "rejects too few generation settings",
      "File:model.json|once upon a time|0.8 0.9",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.settingsCount,
    ],
    [
      "rejects too many generation settings",
      "File:model.json|once upon a time|0.8 0.9 3 extra",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.settingsCount,
    ],
  ])("%s", (_description, input, expectedMessage) => {
    const plan = requireValidationPlan(input);

    expect(plan).toStrictEqual({
      kind: "validation",
      assistantContent: expectedMessage,
    });

    expect(plan.assistantContent).toContain(TRANSFORMER_FILE_COMMAND_USAGE);

    expect(plan).not.toHaveProperty("endpoint");
    expect(plan).not.toHaveProperty("body");
    expect(plan).not.toHaveProperty("mode");
  });
});

describe("saved Transformer number validation", () => {
  it.each([
    ["rejects nonnumeric text", "File:model.json|prompt|hot 0.9 3"],
    ["rejects Boolean-like text", "File:model.json|prompt|true 0.9 3"],
    ["rejects NaN text", "File:model.json|prompt|NaN 0.9 3"],
    ["rejects positive infinity text", "File:model.json|prompt|Infinity 0.9 3"],
    [
      "rejects negative infinity text",
      "File:model.json|prompt|-Infinity 0.9 3",
    ],
    [
      "rejects a finite-looking value that overflows",
      "File:model.json|prompt|1e309 0.9 3",
    ],
    [
      "rejects trailing junk after a numeric prefix",
      "File:model.json|prompt|0.8junk 0.9 3",
    ],
  ])("%s", (_description, input) => {
    const plan = requireValidationPlan(input);

    expect(plan.assistantContent).toBe(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.numeric,
    );

    expect(plan).not.toHaveProperty("endpoint");
    expect(plan).not.toHaveProperty("body");
  });

  it.each([
    [
      "rejects temperature immediately below the lower bound",
      "File:model.json|prompt|0.099999 0.9 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.temperatureRange,
    ],
    [
      "rejects temperature immediately above the upper bound",
      "File:model.json|prompt|2.000001 0.9 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.temperatureRange,
    ],
    [
      "rejects top-p immediately below the lower bound",
      "File:model.json|prompt|0.8 0.099999 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.topPRange,
    ],
    [
      "rejects top-p immediately above the upper bound",
      "File:model.json|prompt|0.8 1.000001 3",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.topPRange,
    ],
    [
      "rejects maximum tokens below the lower bound",
      "File:model.json|prompt|0.8 0.9 2",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.maxTokensRange,
    ],
    [
      "rejects maximum tokens above the upper bound",
      "File:model.json|prompt|0.8 0.9 501",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.maxTokensRange,
    ],
    [
      "rejects fractional maximum tokens",
      "File:model.json|prompt|0.8 0.9 3.5",
      TRANSFORMER_FILE_VALIDATION_MESSAGES.maxTokensInteger,
    ],
  ])("%s", (_description, input, expectedMessage) => {
    const plan = requireValidationPlan(input);

    expect(plan.assistantContent).toBe(expectedMessage);
    expect(plan).not.toHaveProperty("endpoint");
    expect(plan).not.toHaveProperty("body");
  });

  it.each([
    ["accepts the temperature lower bound", "0.1 0.9 3", 0.1, 0.9, 3],
    ["accepts the temperature upper bound", "2.0 0.9 3", 2, 0.9, 3],
    ["accepts the top-p lower bound", "0.8 0.1 3", 0.8, 0.1, 3],
    ["accepts the top-p upper bound", "0.8 1.0 3", 0.8, 1, 3],
    ["accepts the maximum-token lower bound", "0.8 0.9 3", 0.8, 0.9, 3],
    ["accepts the maximum-token upper bound", "0.8 0.9 500", 0.8, 0.9, 500],
  ])(
    "%s",
    (
      _description,
      settings,
      expectedTemperature,
      expectedTopP,
      expectedMaxTokens,
    ) => {
      const plan = requireLoadPlan(`File:model.json|prompt|${settings}`);

      expect(plan.body).toStrictEqual({
        modelFile: "model.json",
        prompt: "prompt",
        temperature: expectedTemperature,
        topP: expectedTopP,
        maxTokens: expectedMaxTokens,
      });
    },
  );
});

describe("Transformer request-start message replacement", () => {
  const previousMessages: Message[] = [
    {
      id: "old-user",
      role: "user",
      content: "old command",
    },
    {
      id: "old-assistant",
      role: "assistant",
      content: "old result",
    },
  ];

  it("discards prior messages for a valid training request", () => {
    const plan = requireTrainingPlan("50 1.0 0.6 1 3");

    const userMessage: Message = {
      id: "new-training-user",
      role: "user",
      content: "50 1.0 0.6 1 3",
    };

    const assistantMessage: Message = {
      id: "new-training-assistant",
      role: "assistant",
      content: "",
    };

    expect(plan.mode).toBe("training");

    expect(
      replaceTransformerMessages(
        previousMessages,
        userMessage,
        assistantMessage,
      ),
    ).toStrictEqual([userMessage, assistantMessage]);
  });

  it("discards prior messages for a valid load request", () => {
    const command = "File:model.json|once upon a time|0.8 0.9 3";

    const plan = requireLoadPlan(command);

    const userMessage: Message = {
      id: "new-load-user",
      role: "user",
      content: command,
    };

    const assistantMessage: Message = {
      id: "new-load-assistant",
      role: "assistant",
      content: "",
    };

    expect(plan.mode).toBe("load");

    expect(
      replaceTransformerMessages(
        previousMessages,
        userMessage,
        assistantMessage,
      ),
    ).toStrictEqual([userMessage, assistantMessage]);
  });

  it("replaces stale output with the command and local validation text", () => {
    const command = "File:model.json||0.8 0.9 3";
    const plan = requireValidationPlan(command);

    const userMessage: Message = {
      id: "new-invalid-user",
      role: "user",
      content: command,
    };

    const assistantMessage: Message = {
      id: "new-invalid-assistant",
      role: "assistant",
      content: plan.assistantContent,
    };

    expect(plan).not.toHaveProperty("endpoint");
    expect(plan).not.toHaveProperty("body");

    expect(
      replaceTransformerMessages(
        previousMessages,
        userMessage,
        assistantMessage,
      ),
    ).toStrictEqual([userMessage, assistantMessage]);

    expect(assistantMessage.content).toBe(
      TRANSFORMER_FILE_VALIDATION_MESSAGES.prompt,
    );
  });
});
