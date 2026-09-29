import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = process.env.VITE_API ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: api, changeOrigin: true, rewrite: (p) => p.replace(/^\/api/, "") },
      "/ws": { target: api.replace("http", "ws"), ws: true },
      "/reports": { target: api, changeOrigin: true },
    },
  },
});
