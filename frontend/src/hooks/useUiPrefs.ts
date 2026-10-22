import { useCallback, useEffect, useRef, useState } from 'react'
import { loadPrefs, savePrefs, type UiPrefs } from '@/features/monitoring/prefs'

/** Preferências de interface persistidas em localStorage (largura/recolhimento das barras, layout, overlays, ordem). */
export function useUiPrefs(): [UiPrefs, (fn: (p: UiPrefs) => UiPrefs) => void] {
  const [prefs, setPrefs] = useState<UiPrefs>(() => loadPrefs())
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const latest = useRef(prefs)

  useEffect(() => {
    latest.current = prefs
    // grava com atraso: arrastar uma barra de redimensionamento gera muitos eventos
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => savePrefs(latest.current), 150)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [prefs])

  useEffect(() => {
    const flush = () => savePrefs(latest.current)
    window.addEventListener('pagehide', flush)
    return () => window.removeEventListener('pagehide', flush)
  }, [])

  const update = useCallback((fn: (p: UiPrefs) => UiPrefs) => setPrefs((p) => fn(p)), [])
  return [prefs, update]
}
