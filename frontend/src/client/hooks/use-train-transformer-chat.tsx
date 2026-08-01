/**
 * Hook for Transformer training and saved-model command routing.
 *
 * Numeric commands continue to start fresh Transformer training.
 * Commands beginning with File: start saved-model generation requests.
 * Named SSE envelopes are reduced into separate training and saved-model
 * display-state branches before rendering in the existing assistant area.
 */
import type {
  TransformerDisplayState,
  TransformerSSEEnvelope,
} from "../lib/transformer-event-state.js";

import {
  SavedTransformerResult,
  TrainTransformerResult,
} from "../components/train-transformer-result/index.js";
import {
  planTransformerSubmission,
  replaceTransformerMessages,
} from "../lib/transformer-command.js";
import {
  createInitialTransformerDisplayState,
  reduceTransformerEvent,
} from "../lib/transformer-event-state.js";
import { useSSEChat } from "./use-sse-chat.js";

type TransformerDisplayStateHolder = {
  current: TransformerDisplayState;
};

export function useTrainTransformerChat() {
  return useSSEChat<TransformerDisplayStateHolder, TransformerSSEEnvelope>({
    title: "Train Transformer",
    tagline:
      "train a GPT from scratch — try: 300 0.8 0.9 2 40 " +
      "(epochs, temp, top-p, layers, max tokens)",
    prepareSubmission: planTransformerSubmission,
    startMessages: replaceTransformerMessages,
    mode: "json-envelope",

    initState: () => ({
      current: createInitialTransformerDisplayState(),
    }),

    onEvent: (envelope, state) => {
      const previousDisplayState = state.current;

      const nextDisplayState = reduceTransformerEvent(
        previousDisplayState,
        envelope,
      );

      if (nextDisplayState === previousDisplayState) {
        return;
      }

      state.current = nextDisplayState;

      if (nextDisplayState.kind === "training") {
        return (
          <TrainTransformerResult
            init={nextDisplayState.init}
            epochs={[...nextDisplayState.epochs]}
            samples={[...nextDisplayState.samples]}
            summary={nextDisplayState.summary}
          />
        );
      }

      if (nextDisplayState.kind === "saved-model") {
        return <SavedTransformerResult state={nextDisplayState} />;
      }
    },
  });
}
