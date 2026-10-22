/** Configuração em tempo de execução (config.js gerado no start do contêiner a partir de variáveis de ambiente). */
export interface AppConfig {
  API_BASE_URL: string
  ENVIRONMENT: string
}

declare global {
  interface Window {
    __APP_CONFIG__?: Partial<AppConfig>
  }
}

export function getConfig(): AppConfig {
  const c = (typeof window !== 'undefined' && window.__APP_CONFIG__) || {}
  return { API_BASE_URL: c.API_BASE_URL || '/api', ENVIRONMENT: c.ENVIRONMENT || 'local' }
}

/** URL de WebSocket a partir da base da API (relativa ou absoluta). */
export function wsUrl(path: string, base = getConfig().API_BASE_URL): string {
  const abs = new URL(`${base.replace(/\/$/, '')}${path}`, window.location.href)
  abs.protocol = abs.protocol === 'https:' ? 'wss:' : 'ws:'
  return abs.toString()
}
