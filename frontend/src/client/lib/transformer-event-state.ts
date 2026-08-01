import type { SSEJSONEnvelope } from "./sse.js";

const TRAINING_INIT_KEYS = [
  "vocabSize",
  "contextLen",
  "embeddingDim",
  "numHeads",
  "ffDim",
  "numLayers",
  "totalParams",
  "temperature",
  "topP",
  "corpusSentences",
  "trainingSequences",
] as const;

const TRAINING_EPOCH_KEYS = ["epoch", "loss", "sample"] as const;

const TRAINING_DONE_KEYS = ["architecture", "finalLoss", "samples"] as const;

const TRAINING_SAMPLE_KEYS = ["epoch", "text"] as const;

const SAVED_TRANSFORMER_LOADED_KEYS = ["file", "prompt"] as const;

const SAVED_TRANSFORMER_RESULT_KEYS = ["text"] as const;

const SAVED_TRANSFORMER_ERROR_KEYS = ["error"] as const;

export type TransformerSSEEnvelope = SSEJSONEnvelope<unknown>;

export type TransformerTrainingInit = {
  vocabSize: number;
  contextLen: number;
  embeddingDim: number;
  numHeads: number;
  ffDim: number;
  numLayers: number;
  totalParams: number;
  temperature: number;
  topP: number;
  corpusSentences: number;
  trainingSequences: number;
};

export type TransformerTrainingEpoch = {
  epoch: number;
  loss: number;
  sample: string;
};

export type TransformerTrainingSample = {
  epoch: number;
  text: string;
};

export type TransformerTrainingSummary = {
  architecture: string;
  finalLoss: number;
};

export type TransformerTrainingDonePayload = {
  architecture: string;
  finalLoss: number;
  samples: TransformerTrainingSample[];
};

export type SavedTransformerLoadedPayload = {
  file: string;
  prompt: string;
};

export type SavedTransformerResultPayload = {
  text: string;
};

export type SavedTransformerErrorPayload = {
  error: string;
};

export type TransformerEmptyDisplayState = {
  kind: "empty";
};

export type TransformerTrainingDisplayState = {
  kind: "training";
  init?: TransformerTrainingInit;
  epochs: readonly TransformerTrainingEpoch[];
  samples: readonly TransformerTrainingSample[];
  summary?: TransformerTrainingSummary;
};

export type SavedTransformerLoadedDisplayState = {
  kind: "saved-model";
  status: "loaded";
  file: string;
  prompt: string;
  text?: string;
};

export type SavedTransformerErrorDisplayState = {
  kind: "saved-model";
  status: "error";
  error: string;
};

export type SavedTransformerDisplayState =
  | SavedTransformerLoadedDisplayState
  | SavedTransformerErrorDisplayState;

export type TransformerDisplayState =
  | TransformerEmptyDisplayState
  | TransformerTrainingDisplayState
  | SavedTransformerDisplayState;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function hasExactKeys(
  value: Record<string, unknown>,
  expectedKeys: readonly string[],
): boolean {
  const actualKeys = Object.keys(value);

  return (
    actualKeys.length === expectedKeys.length &&
    expectedKeys.every((key) =>
      Object.prototype.hasOwnProperty.call(value, key),
    )
  );
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isFiniteInteger(value: unknown): value is number {
  return isFiniteNumber(value) && Number.isInteger(value);
}

function isTransformerTrainingSample(
  value: unknown,
): value is TransformerTrainingSample {
  if (!isRecord(value) || !hasExactKeys(value, TRAINING_SAMPLE_KEYS)) {
    return false;
  }

  return isFiniteInteger(value.epoch) && typeof value.text === "string";
}

function isTransformerTrainingInit(
  value: unknown,
): value is TransformerTrainingInit {
  if (!isRecord(value) || !hasExactKeys(value, TRAINING_INIT_KEYS)) {
    return false;
  }

  return (
    isFiniteInteger(value.vocabSize) &&
    isFiniteInteger(value.contextLen) &&
    isFiniteInteger(value.embeddingDim) &&
    isFiniteInteger(value.numHeads) &&
    isFiniteInteger(value.ffDim) &&
    isFiniteInteger(value.numLayers) &&
    isFiniteInteger(value.totalParams) &&
    isFiniteNumber(value.temperature) &&
    isFiniteNumber(value.topP) &&
    isFiniteInteger(value.corpusSentences) &&
    isFiniteInteger(value.trainingSequences)
  );
}

function isTransformerTrainingEpoch(
  value: unknown,
): value is TransformerTrainingEpoch {
  if (!isRecord(value) || !hasExactKeys(value, TRAINING_EPOCH_KEYS)) {
    return false;
  }

  return (
    isFiniteInteger(value.epoch) &&
    isFiniteNumber(value.loss) &&
    typeof value.sample === "string"
  );
}

function isTransformerTrainingDone(
  value: unknown,
): value is TransformerTrainingDonePayload {
  if (!isRecord(value) || !hasExactKeys(value, TRAINING_DONE_KEYS)) {
    return false;
  }

  return (
    typeof value.architecture === "string" &&
    isFiniteNumber(value.finalLoss) &&
    Array.isArray(value.samples) &&
    value.samples.every(isTransformerTrainingSample)
  );
}

function isSavedTransformerLoaded(
  value: unknown,
): value is SavedTransformerLoadedPayload {
  if (!isRecord(value) || !hasExactKeys(value, SAVED_TRANSFORMER_LOADED_KEYS)) {
    return false;
  }

  return typeof value.file === "string" && typeof value.prompt === "string";
}

function isSavedTransformerResult(
  value: unknown,
): value is SavedTransformerResultPayload {
  if (!isRecord(value) || !hasExactKeys(value, SAVED_TRANSFORMER_RESULT_KEYS)) {
    return false;
  }

  return typeof value.text === "string";
}

function isSavedTransformerDone(value: unknown): boolean {
  return isRecord(value) && Object.keys(value).length === 0;
}

function isSavedTransformerError(
  value: unknown,
): value is SavedTransformerErrorPayload {
  if (!isRecord(value) || !hasExactKeys(value, SAVED_TRANSFORMER_ERROR_KEYS)) {
    return false;
  }

  return typeof value.error === "string";
}

export function createInitialTransformerDisplayState(): TransformerEmptyDisplayState {
  return {
    kind: "empty",
  };
}

export function reduceTransformerEvent(
  state: TransformerDisplayState,
  envelope: TransformerSSEEnvelope,
): TransformerDisplayState {
  switch (envelope.event) {
    case "init": {
      if (state.kind !== "empty" || !isTransformerTrainingInit(envelope.data)) {
        return state;
      }

      return {
        kind: "training",
        init: {
          ...envelope.data,
        },
        epochs: [],
        samples: [],
      };
    }

    case "epoch": {
      if (
        state.kind !== "training" ||
        !isTransformerTrainingEpoch(envelope.data)
      ) {
        return state;
      }

      const nextSamples =
        envelope.data.sample.length > 0
          ? [
              ...state.samples,
              {
                epoch: envelope.data.epoch,
                text: envelope.data.sample,
              },
            ]
          : state.samples;

      return {
        ...state,
        epochs: [
          ...state.epochs,
          {
            ...envelope.data,
          },
        ],
        samples: nextSamples,
      };
    }

    case "loaded": {
      if (state.kind !== "empty" || !isSavedTransformerLoaded(envelope.data)) {
        return state;
      }

      return {
        kind: "saved-model",
        status: "loaded",
        file: envelope.data.file,
        prompt: envelope.data.prompt,
      };
    }

    case "result": {
      if (
        state.kind !== "saved-model" ||
        state.status !== "loaded" ||
        !isSavedTransformerResult(envelope.data)
      ) {
        return state;
      }

      return {
        ...state,
        text: envelope.data.text,
      };
    }

    case "done": {
      if (
        state.kind === "training" &&
        isTransformerTrainingDone(envelope.data)
      ) {
        return {
          ...state,
          samples: envelope.data.samples.map((sample) => ({
            ...sample,
          })),
          summary: {
            architecture: envelope.data.architecture,
            finalLoss: envelope.data.finalLoss,
          },
        };
      }

      if (
        state.kind === "saved-model" &&
        isSavedTransformerDone(envelope.data)
      ) {
        return state;
      }

      return state;
    }

    case "error": {
      if (!isSavedTransformerError(envelope.data)) {
        return state;
      }

      return {
        kind: "saved-model",
        status: "error",
        error: envelope.data.error,
      };
    }

    default:
      return state;
  }
}
