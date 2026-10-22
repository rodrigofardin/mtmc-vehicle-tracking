import { defineConfig } from '@playwright/test'

// E2E contra o stack REAL (docker compose up -d). Não há backend simulado.
//   E2E_BASE_URL  (padrão http://localhost:8088)
//   E2E_BROWSER_CHANNEL (padrão msedge; use "chrome" ou deixe vazio para o Chromium do Playwright)
export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  retries: 0,
  reporter: 'list',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://localhost:8088',
    channel: process.env.E2E_BROWSER_CHANNEL ?? 'msedge',
    viewport: { width: 1600, height: 900 },
    trace: 'retain-on-failure',
  },
})
