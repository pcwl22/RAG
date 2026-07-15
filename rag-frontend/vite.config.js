import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [vue()],
  server: {
    // Keep browser requests same-origin in development. Production should use
    // the same /api and /health proxy contract and inject credentials there.
    proxy: {
      '/api': {
        target: process.env.RAG_API_UPSTREAM || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      '/health': {
        target: process.env.RAG_API_UPSTREAM || 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})
