import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: '/3d-test/',
  server: {
    port: 5174,
    proxy: {
      '/3d-test-api': {
        target: 'http://elohim-bitch-gpu.insanelabs.org',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/3d-test-api/, ''),
      },
    },
  },
})
