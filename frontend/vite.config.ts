import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The built app is served by the FastAPI backend at /app (see codesync/api.py).
// In development Vite serves it and proxies the API paths to the backend.
const API = "http://127.0.0.1:8000";

export default defineConfig({
  base: "/app/",
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      ["/documents", "/comparisons", "/jobs", "/health"].map((p) => [p, { target: API, changeOrigin: true }]),
    ),
  },
});
