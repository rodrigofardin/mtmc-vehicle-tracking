import { useEffect, useState, type RefObject } from 'react'

/** Largura e altura úteis de um elemento (ResizeObserver). Valores iniciais razoáveis para o primeiro render/jsdom. */
export function useElementSize(ref: RefObject<Element | null>, initial = { width: 1200, height: 700 }) {
  const [size, setSize] = useState(initial)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof ResizeObserver === 'undefined') return
    let raf = 0
    const ro = new ResizeObserver(([e]) => {
      cancelAnimationFrame(raf)
      raf = requestAnimationFrame(() => {
        const { width, height } = e.contentRect
        setSize((s) => (Math.abs(s.width - width) < 1 && Math.abs(s.height - height) < 1 ? s : { width, height }))
      })
    })
    ro.observe(el)
    return () => {
      cancelAnimationFrame(raf)
      ro.disconnect()
    }
  }, [ref])
  return size
}
