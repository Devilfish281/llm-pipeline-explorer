// frontend/src/client/hooks/use-train-embed-chat.tsx
/**
 * Hook for Word2Vec Skip-gram training. Input is comma- or space-separated words to compare.
 * Streams corpus stats, epoch losses, then learned embeddings with neighbors and similarities.
 */
import type {
  Analogy,
  EpochData,
  InitData,
  Neighbor,
  SimilarityPair,
  WordEmbedding,
} from "../components/train-embed-result/index.js";

import { TrainEmbedResult } from "../components/train-embed-result/index.js";
import { useSSEChat } from "./use-sse-chat.js";

const WHITESPACE = /\s+/;

type TrainEmbedState = {
  init?: InitData;
  epochs: EpochData[];
  embeddings?: WordEmbedding[];
  neighbors?: Neighbor[];
  similarities?: SimilarityPair[];
  analogies?: Analogy[];
  warnings?: string[];
};

type DoneEvent = {
  embeddings: WordEmbedding[];
  neighbors: Neighbor[];
  similarities: SimilarityPair[];
  analogies: Analogy[];
  warnings: string[];
};

type TrainEmbedEvent = InitData | EpochData | DoneEvent;

export function useTrainEmbedChat() {
  return useSSEChat<TrainEmbedState, TrainEmbedEvent>({
    endpoint: "/api/train-embed",
    title: "Train Embeddings",
    tagline: "enter: words | epochs dimensions window-size negative-samples",

    buildBody: (input) => {
      const [wordsSection = "", settingsSection = ""] = input.split("|", 2);

      const words = (
        wordsSection.includes(",")
          ? wordsSection.split(",")
          : wordsSection.split(WHITESPACE)
      )
        .map((word) => word.trim().toLowerCase())
        .filter(Boolean);

      const settings = settingsSection.trim().split(WHITESPACE).filter(Boolean);

      const parseBoundedInteger = (
        value: string | undefined,
        fallback: number,
        minimum: number,
        maximum: number,
      ): number => {
        const parsed = Number.parseInt(value ?? "", 10);

        if (!Number.isInteger(parsed) || parsed < minimum || parsed > maximum) {
          return fallback;
        }

        return parsed;
      };

      return {
        words,
        epochs: parseBoundedInteger(settings[0], 10, 10, 10_000),
        dimensions: parseBoundedInteger(settings[1], 4, 4, 64),
        windowSize: parseBoundedInteger(settings[2], 1, 1, 5),
        negativeSamples: parseBoundedInteger(settings[3], 1, 1, 10),
      };
    },

    initState: () => ({
      epochs: [],
    }),

    onEvent: (parsed, state) => {
      if ("vocabSize" in parsed) {
        state.init = parsed as InitData;

        return <TrainEmbedResult init={state.init} epochs={[]} />;
      }

      if ("epoch" in parsed) {
        state.epochs.push(parsed as EpochData);

        return (
          <TrainEmbedResult init={state.init} epochs={[...state.epochs]} />
        );
      }

      if ("embeddings" in parsed) {
        const done = parsed as DoneEvent;

        state.embeddings = done.embeddings;
        state.neighbors = done.neighbors;
        state.similarities = done.similarities;
        state.analogies = done.analogies;
        state.warnings = done.warnings;

        return (
          <TrainEmbedResult
            init={state.init}
            epochs={[...state.epochs]}
            embeddings={state.embeddings}
            neighbors={state.neighbors}
            similarities={state.similarities}
            analogies={state.analogies}
            warnings={state.warnings}
          />
        );
      }
    },
  });
}
