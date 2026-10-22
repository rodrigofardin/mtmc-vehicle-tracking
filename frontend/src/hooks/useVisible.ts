import { useEffect, useState, type RefObject } from 'react'

/** Verdadeiro se o elemento está na viewport E a aba está visível (usado para pausar fluxos não vistos). */
export function useVisible(ref: RefObject<Element | null>): boolean {
  const [inView, setInView] = useState(true)
  const [tabVisible, setTabVisible] = useState(() => typeof document === 'undefined' || !document.hidden)

  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0.01 })
    io.observe(el)
    return () => io.disconnect()
  }, [ref])

  useEffect(() => {
    const on = () => setTabVisible(!document.hidden)
    document.addEventListener('visibilitychange', on)
    return () => document.removeEventListener('visibilitychange', on)
  }, [])

  return inView && tabVisible
}
