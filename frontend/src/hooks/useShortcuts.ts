import { useEffect, useRef } from 'react'

/** Verdadeiro se o foco está num campo onde o usuário digita (os atalhos NÃO podem interferir). */
export function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false
  if (el.isContentEditable) return true
  const tag = el.tagName
  if (tag === 'INPUT') {
    const t = (el as HTMLInputElement).type
    return !['checkbox', 'radio', 'button', 'submit', 'range'].includes(t)
  }
  return tag === 'TEXTAREA' || tag === 'SELECT'
}

/** Há diálogo modal aberto (confirmação etc.)? Atalhos globais ficam suspensos. */
export function modalOpen(): boolean {
  return typeof document !== 'undefined' && !!document.querySelector('[role="alertdialog"], [role="dialog"][data-state="open"]')
}

export type ShortcutMap = Record<string, () => void>

/**
 * Atalhos globais de uma tecla (sem Ctrl/Alt/Meta). Ignorados ao digitar em campos, em seletores/abas focados com
 * navegação própria (setas) e com diálogo aberto. `Escape` é tratado fora daqui, pelas telas que o usam.
 */
export function useShortcuts(map: ShortcutMap, enabled = true) {
  const ref = useRef(map)
  useEffect(() => {
    ref.current = map
  })
  useEffect(() => {
    if (!enabled) return
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.altKey || e.metaKey || e.defaultPrevented) return
      if (isTypingTarget(e.target) || modalOpen()) return
      const fn = ref.current[e.key.length === 1 ? e.key.toLowerCase() : e.key]
      if (fn) {
        e.preventDefault()
        fn()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [enabled])
}
