import type {
  SavedTransformerErrorDisplayState,
  SavedTransformerLoadedDisplayState,
  TransformerDisplayState,
  TransformerSSEEnvelope,
  TransformerTrainingDisplayState,
  TransformerTrainingDonePayload,
  TransformerTrainingEpoch,
  TransformerTrainingInit,
} from "./transformer-event-state.js";

import { describe, expect, it } from "vitest";

import { planTransformerSubmission } from "./transformer-command.js";
import {
  createInitialTransformerDisplayState,
  reduceTransformerEvent,
} from "./transformer-event-state.js";

const TRAINING_INIT: TransformerTrainingInit = {
  vocabSize: 192,
  contextLen: 32,
  embeddingDim: 32,
  numHeads: 2,
  ffDim: 128,
  numLayers: 1,
  totalParams: 39_272,
  temperature: 1,
  topP: 0.6,
  corpusSentences: 107,
  trainingSequences: 2_092,
};

const FIRST_TRAINING_EPOCH: TransformerTrainingEpoch = {
  epoch: 0,
  loss: 4.123456,
  sample: "Transformer worker processes: 3\n\n" + "once upon a tall king",
};

const TRAINING_DONE: TransformerTrainingDonePayload = {
  architecture: "Decoder-Only Transformer (1 layers, 32d, 2h, 128ff)",
  finalLoss: 3.7488,
  samples: [
    {
      epoch: 0,
      text: "once upon a tall king",
    },
    {
      epoch: 50,
      text: "once upon a time went home",
    },
  ],
};

const SAFE_TRANSFORMER_ERROR_CASES = [
  ["named-model loading", "The saved Transformer model could not be loaded."],
  ["latest-model absence", "No valid saved Transformer model was found."],
  ["empty prompt", "The prompt must not be empty."],
  [
    "unsupported prompt",
    "The prompt contains text that this saved Transformer model cannot tokenize.",
  ],
  ["overlength prompt", "The prompt must contain no more than 16 tokens."],
  [
    "generation failure",
    "The saved Transformer model could not generate text.",
  ],
  [
    "generation deadline",
    "Saved Transformer generation exceeded its time limit.",
  ],
] as const;

function envelope(event: string, data: unknown): TransformerSSEEnvelope {
  return {
    event,
    data,
  };
}

function reduceEvents(
  events: readonly TransformerSSEEnvelope[],
): TransformerDisplayState {
  let state: TransformerDisplayState = createInitialTransformerDisplayState();

  for (const event of events) {
    state = reduceTransformerEvent(state, event);
  }

  return state;
}

function requireTrainingState(
  state: TransformerDisplayState,
): TransformerTrainingDisplayState {
  if (state.kind !== "training") {
    throw new Error(
      `Expected training state, received ${JSON.stringify(state)}.`,
    );
  }

  return state;
}

function requireSavedLoadedState(
  state: TransformerDisplayState,
): SavedTransformerLoadedDisplayState {
  if (state.kind !== "saved-model" || state.status !== "loaded") {
    throw new Error(
      `Expected loaded saved-model state, received ${JSON.stringify(state)}.`,
    );
  }

  return state;
}

function requireSavedErrorState(
  state: TransformerDisplayState,
): SavedTransformerErrorDisplayState {
  if (state.kind !== "saved-model" || state.status !== "error") {
    throw new Error(
      `Expected saved-model error state, received ${JSON.stringify(state)}.`,
    );
  }

  return state;
}

describe("Transformer training event state", () => {
  it("reduces the complete init, epoch, and done progression", () => {
    const afterInit = reduceTransformerEvent(
      createInitialTransformerDisplayState(),
      envelope("init", TRAINING_INIT),
    );

    expect(afterInit).toStrictEqual({
      kind: "training",
      init: TRAINING_INIT,
      epochs: [],
      samples: [],
    });

    const afterEpoch = reduceTransformerEvent(
      afterInit,
      envelope("epoch", FIRST_TRAINING_EPOCH),
    );

    expect(afterEpoch).toStrictEqual({
      kind: "training",
      init: TRAINING_INIT,
      epochs: [FIRST_TRAINING_EPOCH],
      samples: [
        {
          epoch: 0,
          text: FIRST_TRAINING_EPOCH.sample,
        },
      ],
    });

    const afterDone = reduceTransformerEvent(
      afterEpoch,
      envelope("done", TRAINING_DONE),
    );

    expect(afterDone).toStrictEqual({
      kind: "training",
      init: TRAINING_INIT,
      epochs: [FIRST_TRAINING_EPOCH],
      samples: TRAINING_DONE.samples,
      summary: {
        architecture: TRAINING_DONE.architecture,
        finalLoss: TRAINING_DONE.finalLoss,
      },
    });
  });

  it("preserves the first worker-process sample byte-for-byte", () => {
    const state = requireTrainingState(
      reduceEvents([
        envelope("init", TRAINING_INIT),
        envelope("epoch", FIRST_TRAINING_EPOCH),
      ]),
    );

    expect(state.epochs[0]?.sample).toBe(
      "Transformer worker processes: 3\n\n" + "once upon a tall king",
    );

    expect(state.samples[0]?.text).toBe(FIRST_TRAINING_EPOCH.sample);

    expect(
      state.samples[0]?.text.match(/Transformer worker processes:/g),
    ).toHaveLength(1);
  });

  it("preserves later training samples without adding or repeating the worker label", () => {
    const laterEpoch: TransformerTrainingEpoch = {
      epoch: 50,
      loss: 3.75,
      sample: "once upon a time went home",
    };

    const state = requireTrainingState(
      reduceEvents([
        envelope("init", TRAINING_INIT),
        envelope("epoch", FIRST_TRAINING_EPOCH),
        envelope("epoch", laterEpoch),
      ]),
    );

    expect(state.epochs[1]).toStrictEqual(laterEpoch);

    expect(state.samples[1]).toStrictEqual({
      epoch: laterEpoch.epoch,
      text: laterEpoch.sample,
    });

    expect(state.samples[1]?.text).not.toContain(
      "Transformer worker processes:",
    );

    expect(
      state.samples[0]?.text.match(/Transformer worker processes:/g),
    ).toHaveLength(1);
  });

  it("uses the backend done samples without adding the worker label", () => {
    const state = requireTrainingState(
      reduceEvents([
        envelope("init", TRAINING_INIT),
        envelope("epoch", FIRST_TRAINING_EPOCH),
        envelope("done", TRAINING_DONE),
      ]),
    );

    expect(state.samples).toStrictEqual(TRAINING_DONE.samples);

    expect(JSON.stringify(state.samples)).not.toContain(
      "Transformer worker processes:",
    );
  });
});

describe("Saved Transformer event state", () => {
  it("creates only filename and prompt state from loaded", () => {
    const state = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "transformer-weights-e50-l1-d32-h2-ff128-ctx32.json",
          prompt: "once upon a time",
        }),
      ]),
    );

    expect(state).toStrictEqual({
      kind: "saved-model",
      status: "loaded",
      file: "transformer-weights-e50-l1-d32-h2-ff128-ctx32.json",
      prompt: "once upon a time",
    });

    expect(Object.keys(state).sort()).toStrictEqual([
      "file",
      "kind",
      "prompt",
      "status",
    ]);

    expect(state).not.toHaveProperty("epochs");
    expect(state).not.toHaveProperty("samples");
    expect(state).not.toHaveProperty("summary");
    expect(state).not.toHaveProperty("workerCount");
  });

  it("preserves the returned prompt without trimming it again", () => {
    const returnedPrompt = "  once   upon\ta time  ";

    const state = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "model.json",
          prompt: returnedPrompt,
        }),
      ]),
    );

    expect(state.prompt).toBe(returnedPrompt);
  });

  it("adds one complete result string without splitting it", () => {
    const completeText =
      "once upon a time went you will\n" + "and then returned home";

    const state = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "model.json",
          prompt: "once upon a time",
        }),
        envelope("result", {
          text: completeText,
        }),
      ]),
    );

    expect(state.text).toBe(completeText);
    expect(typeof state.text).toBe("string");
    expect(Array.isArray(state.text)).toBe(false);
  });

  it("keeps completed saved-model state limited to renderer-approved fields", () => {
    const state = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "model.json",
          prompt: "once upon a time",
        }),
        envelope("result", {
          text: "once upon a time went home",
        }),
      ]),
    );

    expect(Object.keys(state).sort()).toStrictEqual([
      "file",
      "kind",
      "prompt",
      "status",
      "text",
    ]);

    expect(state).not.toHaveProperty("error");
    expect(state).not.toHaveProperty("init");
    expect(state).not.toHaveProperty("epochs");
    expect(state).not.toHaveProperty("samples");
    expect(state).not.toHaveProperty("summary");
    expect(state).not.toHaveProperty("workerCount");
    expect(state).not.toHaveProperty("workerProcesses");

    expect(JSON.stringify(state)).not.toContain(
      "Transformer worker processes:",
    );
  });

  it("makes load done an invisible state transition", () => {
    const beforeDone = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "model.json",
          prompt: "once upon a time",
        }),
        envelope("result", {
          text: "once upon a time went home",
        }),
      ]),
    );

    const afterDone = reduceTransformerEvent(beforeDone, envelope("done", {}));

    expect(afterDone).toBe(beforeDone);
    expect(afterDone).toStrictEqual(beforeDone);
  });

  it("uses loaded.file as the display filename after a latest request", () => {
    const plan = planTransformerSubmission("File:|once upon a time|0.8 0.9 3");

    if (plan.kind !== "request" || plan.mode !== "load") {
      throw new Error(
        `Expected a saved-model load request, received ${JSON.stringify(plan)}.`,
      );
    }

    expect(plan.body.modelFile).toBeNull();

    const actualSelectedFilename =
      "transformer-weights-e300-l2-d32-h2-ff128-ctx32.json";

    const state = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: actualSelectedFilename,
          prompt: plan.body.prompt,
        }),
      ]),
    );

    expect(state).toStrictEqual({
      kind: "saved-model",
      status: "loaded",
      file: actualSelectedFilename,
      prompt: "once upon a time",
    });

    expect(state.file).toBe(actualSelectedFilename);
    expect(state.file).not.toBe(String(plan.body.modelFile));
    expect(state.file).not.toBe("latest");
  });

  it("does not synthesize a loading worker label or field", () => {
    const state = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "model.json",
          prompt: "prompt",
        }),
        envelope("result", {
          text: "prompt continuation",
        }),
      ]),
    );

    expect(state.text).toBe("prompt continuation");
    expect(state).not.toHaveProperty("workerCount");
    expect(state).not.toHaveProperty("workerProcesses");

    expect(JSON.stringify(state)).not.toContain(
      "Transformer worker processes:",
    );
  });
});

describe("safe Transformer error replacement", () => {
  it.each(SAFE_TRANSFORMER_ERROR_CASES)(
    "replaces empty state with only the exact %s safe message",
    (_caseName, message) => {
      const state = requireSavedErrorState(
        reduceTransformerEvent(
          createInitialTransformerDisplayState(),
          envelope("error", {
            error: message,
          }),
        ),
      );

      expect(state).toStrictEqual({
        kind: "saved-model",
        status: "error",
        error: message,
      });

      expect(Object.keys(state).sort()).toStrictEqual([
        "error",
        "kind",
        "status",
      ]);

      expect(state.error).toBe(message);
      expect(state).not.toHaveProperty("file");
      expect(state).not.toHaveProperty("prompt");
      expect(state).not.toHaveProperty("text");
      expect(state).not.toHaveProperty("epochs");
      expect(state).not.toHaveProperty("samples");
      expect(state).not.toHaveProperty("summary");
    },
  );

  it("replaces loaded filename and prompt with only the safe error", () => {
    const loadedState = reduceEvents([
      envelope("loaded", {
        file: "model.json",
        prompt: "once upon a time",
      }),
    ]);

    const errorState = requireSavedErrorState(
      reduceTransformerEvent(
        loadedState,
        envelope("error", {
          error: "The saved Transformer model could not generate text.",
        }),
      ),
    );

    expect(errorState).toStrictEqual({
      kind: "saved-model",
      status: "error",
      error: "The saved Transformer model could not generate text.",
    });

    expect(errorState).not.toHaveProperty("file");
    expect(errorState).not.toHaveProperty("prompt");
    expect(errorState).not.toHaveProperty("text");
  });

  it("replaces loaded and result data with only the safe error", () => {
    const successfulState = reduceEvents([
      envelope("loaded", {
        file: "model.json",
        prompt: "once upon a time",
      }),
      envelope("result", {
        text: "once upon a time went home",
      }),
    ]);

    const errorState = requireSavedErrorState(
      reduceTransformerEvent(
        successfulState,
        envelope("error", {
          error: "The saved Transformer model could not be loaded.",
        }),
      ),
    );

    expect(errorState).toStrictEqual({
      kind: "saved-model",
      status: "error",
      error: "The saved Transformer model could not be loaded.",
    });

    expect(Object.keys(errorState).sort()).toStrictEqual([
      "error",
      "kind",
      "status",
    ]);

    expect(errorState).not.toHaveProperty("file");
    expect(errorState).not.toHaveProperty("prompt");
    expect(errorState).not.toHaveProperty("text");
  });

  it("replaces all prior training data with only the safe error", () => {
    const trainingState = reduceEvents([
      envelope("init", TRAINING_INIT),
      envelope("epoch", FIRST_TRAINING_EPOCH),
      envelope("done", TRAINING_DONE),
    ]);

    const errorState = requireSavedErrorState(
      reduceTransformerEvent(
        trainingState,
        envelope("error", {
          error: "Saved Transformer generation exceeded its time limit.",
        }),
      ),
    );

    expect(errorState).toStrictEqual({
      kind: "saved-model",
      status: "error",
      error: "Saved Transformer generation exceeded its time limit.",
    });

    expect(errorState).not.toHaveProperty("init");
    expect(errorState).not.toHaveProperty("epochs");
    expect(errorState).not.toHaveProperty("samples");
    expect(errorState).not.toHaveProperty("summary");
  });
});

describe("exact Transformer payload guards", () => {
  const trainingState = reduceEvents([envelope("init", TRAINING_INIT)]);

  const savedState = reduceEvents([
    envelope("loaded", {
      file: "model.json",
      prompt: "prompt",
    }),
  ]);

  const malformedCases: Array<
    [string, TransformerDisplayState, TransformerSSEEnvelope]
  > = [
    [
      "rejects loaded with an extra field",
      createInitialTransformerDisplayState(),
      envelope("loaded", {
        file: "model.json",
        prompt: "prompt",
        path: "private/path",
      }),
    ],
    [
      "rejects loaded with a non-string filename",
      createInitialTransformerDisplayState(),
      envelope("loaded", {
        file: null,
        prompt: "prompt",
      }),
    ],
    [
      "rejects result with an extra field",
      savedState,
      envelope("result", {
        text: "generated text",
        tokens: ["generated", "text"],
      }),
    ],
    [
      "rejects result with a non-string text value",
      savedState,
      envelope("result", {
        text: ["token", "fragments"],
      }),
    ],
    [
      "rejects load done with a nonempty object",
      savedState,
      envelope("done", {
        complete: true,
      }),
    ],
    [
      "rejects error with an extra field",
      savedState,
      envelope("error", {
        error: "Safe message.",
        detail: "private detail",
      }),
    ],
    [
      "rejects training init with a missing field",
      createInitialTransformerDisplayState(),
      envelope("init", {
        ...TRAINING_INIT,
        trainingSequences: undefined,
      }),
    ],
    [
      "rejects training init with an extra field",
      createInitialTransformerDisplayState(),
      envelope("init", {
        ...TRAINING_INIT,
        workerCount: 4,
      }),
    ],
    [
      "rejects training epoch with an extra field",
      trainingState,
      envelope("epoch", {
        ...FIRST_TRAINING_EPOCH,
        workerCount: 4,
      }),
    ],
    [
      "rejects training epoch with a non-string sample",
      trainingState,
      envelope("epoch", {
        epoch: 0,
        loss: 4.123456,
        sample: null,
      }),
    ],
    [
      "rejects training done with an extra field",
      trainingState,
      envelope("done", {
        ...TRAINING_DONE,
        workerCount: 4,
      }),
    ],
    [
      "rejects training done with malformed samples",
      trainingState,
      envelope("done", {
        architecture: TRAINING_DONE.architecture,
        finalLoss: TRAINING_DONE.finalLoss,
        samples: [
          {
            epoch: 50,
            text: "generated text",
            workerCount: 4,
          },
        ],
      }),
    ],
  ];

  it.each(malformedCases)(
    "%s",
    (_description, currentState, malformedEnvelope) => {
      expect(reduceTransformerEvent(currentState, malformedEnvelope)).toBe(
        currentState,
      );
    },
  );

  it("ignores an unknown event without exposing its payload", () => {
    const initialState = createInitialTransformerDisplayState();

    const nextState = reduceTransformerEvent(
      initialState,
      envelope("internal-debug-event", {
        path: "C:\\private\\model.json",
        traceback: "private traceback",
      }),
    );

    expect(nextState).toBe(initialState);
  });
});

describe("training and saved-model branch isolation", () => {
  const trainingState = reduceEvents([envelope("init", TRAINING_INIT)]);

  const savedState = reduceEvents([
    envelope("loaded", {
      file: "model.json",
      prompt: "prompt",
    }),
  ]);

  const crossBranchCases: Array<
    [string, TransformerDisplayState, TransformerSSEEnvelope]
  > = [
    [
      "does not apply a training epoch to saved-model state",
      savedState,
      envelope("epoch", FIRST_TRAINING_EPOCH),
    ],
    [
      "does not apply a training done event to saved-model state",
      savedState,
      envelope("done", TRAINING_DONE),
    ],
    [
      "does not apply a saved-model result to training state",
      trainingState,
      envelope("result", {
        text: "generated text",
      }),
    ],
    [
      "does not interpret empty load done as training completion",
      trainingState,
      envelope("done", {}),
    ],
    [
      "does not switch an active training branch through loaded",
      trainingState,
      envelope("loaded", {
        file: "model.json",
        prompt: "prompt",
      }),
    ],
    [
      "does not switch an active saved-model branch through init",
      savedState,
      envelope("init", TRAINING_INIT),
    ],
  ];

  it.each(crossBranchCases)(
    "%s",
    (_description, currentState, crossBranchEnvelope) => {
      expect(reduceTransformerEvent(currentState, crossBranchEnvelope)).toBe(
        currentState,
      );
    },
  );
});

describe("Transformer stream state isolation", () => {
  it("starts sequential saved-model streams with fresh independent state", () => {
    const firstSuccessfulState = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "first-model.json",
          prompt: "first prompt",
        }),
        envelope("result", {
          text: "first prompt first result",
        }),
      ]),
    );

    const firstErrorState = requireSavedErrorState(
      reduceTransformerEvent(
        firstSuccessfulState,
        envelope("error", {
          error: "The saved Transformer model could not generate text.",
        }),
      ),
    );

    const secondState = requireSavedLoadedState(
      reduceEvents([
        envelope("loaded", {
          file: "second-model.json",
          prompt: "second prompt",
        }),
        envelope("result", {
          text: "second prompt second result",
        }),
      ]),
    );

    expect(firstSuccessfulState).toStrictEqual({
      kind: "saved-model",
      status: "loaded",
      file: "first-model.json",
      prompt: "first prompt",
      text: "first prompt first result",
    });

    expect(firstErrorState).toStrictEqual({
      kind: "saved-model",
      status: "error",
      error: "The saved Transformer model could not generate text.",
    });

    expect(secondState).toStrictEqual({
      kind: "saved-model",
      status: "loaded",
      file: "second-model.json",
      prompt: "second prompt",
      text: "second prompt second result",
    });

    expect(secondState).not.toHaveProperty("error");

    expect(JSON.stringify(secondState)).not.toContain("first-model.json");

    expect(JSON.stringify(secondState)).not.toContain("first prompt");

    expect(JSON.stringify(secondState)).not.toContain("first result");
  });
});
