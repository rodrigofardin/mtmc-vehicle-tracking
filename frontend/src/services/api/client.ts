import { getConfig } from '@/lib/config'
import type { CameraStatus, Health } from '@/types/api'

export class ApiError extends Error {
  readonly status: number | null
  constructor(message: string, status: number | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function getJson<T>(path: string, timeoutMs = 4000, signal?: AbortSignal): Promise<T> {
  const ctrl = new AbortController()
  const timer = setTimeout(() => ctrl.abort(), timeoutMs)
  signal?.addEventListener('abort', () => ctrl.abort())
  try {
    const res = await fetch(`${getConfig().API_BASE_URL}${path}`, { signal: ctrl.signal, cache: 'no-store' })
    if (!res.ok) throw new ApiError(`HTTP ${res.status} em ${path}`, res.status)
    return (await res.json()) as T
  } catch (e) {
    if (e instanceof ApiError) throw e
    if (e instanceof DOMException && e.name === 'AbortError') throw new ApiError(`Tempo esgotado em ${path}`)
    throw new ApiError(`Backend inacessível (${path})`)
  } finally {
    clearTimeout(timer)
  }
}

export const api = {
  health: (signal?: AbortSignal) => getJson<Health>('/health', 4000, signal),
  cameras: (signal?: AbortSignal) => getJson<CameraStatus[]>('/cameras', 4000, signal),
}
