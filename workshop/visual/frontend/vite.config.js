import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  publicDir: '.',
  server: {
    host: true,
    port: 8889,
    open: false,
    proxy: {
      '/api': {
        target: 'http://localhost:8888',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8888',
        ws: true,
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
});
