import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '@/services/api/client'
import type { RuntimeInfo } from '@/types/api'

export interface RuntimeData {
  data: RuntimeInfo | null
  loading: boolean
  error: string | null
  refresh: () => void
}

/**
 * Consulta GET /api/runtime enquanto o painel está aberto. Só leitura: não toca nos streams nem na inferência.
 * Mantém o último dado válido se uma atualização falhar (o erro é exibido junto).
 */
export function useRuntime(enabled: boolean, pollMs = 5000): RuntimeData {
  const [data, setData] = useState<RuntimeInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const tick = useRef<() => void>(() => {})

  useEffect(() => {
    if (!enabled) return
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const ctrl = new AbortController()
    const run = async () => {
      setLoading(true)
      try {
        const r = await api.runtime(ctrl.signal)
        if (!alive) return
        setData(r)
        setError(null)
      } catch (e) {
        if (!alive) return
        setError(e instanceof Error ? e.message : 'Erro desconhecido')
      } finally {
        if (alive) setLoading(false)
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
  }, [enabled, pollMs])

  const refresh = useCallback(() => tick.current(), [])
  return { data, loading, error, refresh }
}
