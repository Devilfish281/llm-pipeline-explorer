import { defineConfig } from "vite";

const backendTarget = "http://127.0.0.1:8000";

export default defineConfig({
  esbuild: {
    jsxImportSource: "hono/jsx/dom",
  },

  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,

    proxy: {
      "/api": {
        target: backendTarget,
        changeOrigin: true,

        // /api/simple-chat becomes /simple-chat.
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
