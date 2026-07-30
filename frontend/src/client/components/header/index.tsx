/** Page header with navigation dropdown (selects the demo page), title, and tagline. */
import type { RoutePath } from "@w3cj/ruta";

import { useLocation } from "@w3cj/ruta";

import { useChatContext } from "../../hooks/use-chat-context.js";
import styles from "./styles.module.css";

const options: { path: RoutePath; label: string }[] = [
  { path: "/", label: "Simple Chat" },
  { path: "/neural-net-xor", label: "XOR Neural Net" },
  { path: "/bpe-token", label: "Basic Tokenizer" },
  { path: "/train-embed", label: "Train Embeddings" },
  { path: "/train-transformer", label: "Train Transformer" },
];

export function Header() {
  const { title, tagline } = useChatContext();
  const { location, navigate } = useLocation();

  return (
    <div class={styles.header}>
      <select
        aria-label="Select a page"
        class={styles.select}
        value={location}
        onChange={(event) => {
          const selectedPath = (event.target as HTMLSelectElement)
            .value as RoutePath;

          navigate(selectedPath);
        }}
      >
        {options.map((route) => (
          <option key={route.path} value={route.path}>
            {route.label}
          </option>
        ))}
      </select>

      <h1 class={styles.title}>{title}</h1>
      <p class={styles.tagline}>{tagline}</p>
    </div>
  );
}
