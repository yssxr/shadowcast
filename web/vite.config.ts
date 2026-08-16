import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Artifacts live in `public/` and are copied verbatim on build.
//
// `data.bin.gz` is pre-compressed. A host that sets `Content-Encoding: gzip` lets the
// browser inflate it during transfer, which is the cheap path; a host that cannot set
// headers, GitHub Pages among them, hands over the raw bytes and the reader inflates them
// itself. Both work, and the second is verified against a plain `python -m http.server`.
//
// `base` is relative so the build runs from any path, including the `/<repo>/` prefix a
// project Pages site is served under.
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { target: "es2022", assetsInlineLimit: 0 },
});
