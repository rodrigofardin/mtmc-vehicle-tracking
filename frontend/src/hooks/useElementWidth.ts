import { useEffect, useState, type RefObject } from 'react'

export function useElementWidth(ref: RefObject<Element | null>): number {
  const [w, setW] = useState(1200)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [ref])
  return w
}
