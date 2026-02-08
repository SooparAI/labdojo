import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      // Proxy API calls to the local FastAPI backend
      '/chat': 'http://localhost:8080',
      '/papers': 'http://localhost:8080',
      '/pipeline': 'http://localhost:8080',
      '/projects': 'http://localhost:8080',
      '/apis': 'http://localhost:8080',
      '/settings': 'http://localhost:8080',
      '/export': 'http://localhost:8080',
      '/status': 'http://localhost:8080',
      '/conversation': 'http://localhost:8080',
      '/learning': 'http://localhost:8080',
    },
  },
})
