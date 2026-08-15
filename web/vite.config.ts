import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Artifacts are served from `public/` in dev and copied verbatim on build. In
// production `data.bin.gz` must go out with `Content-Encoding: gzip` so the browser
// inflates it during transfer — that is why the reader ships no decompression code.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { target: "es2022", assetsInlineLimit: 0 },
});
