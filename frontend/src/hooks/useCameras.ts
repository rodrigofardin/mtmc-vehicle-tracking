import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '@/services/api/client'
import type { CameraStatus, Health } from '@/types/api'

export type BackendConnection = 'loading' | 'online' | 'offline'

export interface CamerasData {
  cameras: CameraStatus[]
  health: Health | null
  connection: BackendConnection
  error: string | null
  refresh: () => void
}

/** Descobre as câmeras dinamicamente e consulta o estado confirmado pelo backend (polling). */
export function useCameras(pollMs = 2000): CamerasData {
  const [cameras, setCameras] = useState<CameraStatus[]>([])
  const [health, setHealth] = useState<Health | null>(null)
  const [connection, setConnection] = useState<BackendConnection>('loading')
  const [error, setError] = useState<string | null>(null)
  const tick = useRef<() => void>(() => {})

  useEffect(() => {
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const ctrl = new AbortController()
    const run = async () => {
      try {
        const [cams, h] = await Promise.all([api.cameras(ctrl.signal), api.health(ctrl.signal)])
        if (!alive) return
        setCameras(cams)
        setHealth(h)
        setConnection('online')
        setError(null)
      } catch (e) {
        if (!alive) return
        setConnection('offline')
        setError(e instanceof Error ? e.message : 'Erro desconhecido')
      }
      if (alive) timer = setTimeout(run, pollMs)
    }
    tick.current = () => {
      clearTimeout(timer)
      void run()
    }
    void run()
    return () => {
      alive = false
      clearTimeout(timer)
      ctrl.abort()
    }
  }, [pollMs])

  const refresh = useCallback(() => tick.current(), [])
  return { cameras, health, connection, error, refresh }
}
