import { useCallback, useEffect, useMemo, useState } from 'react'

const KEY = 'mtmc.selection.v1'

function load(): string[] | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    return raw ? (JSON.parse(raw) as string[]) : null
  } catch {
    return null
  }
}

/**
 * Seleção de câmeras exibidas (persistida só na sessão do navegador).
 * Sem seleção salva => todas as câmeras descobertas. Remover do grid NÃO para a inferência no backend.
 */
export function useSelection(allIds: string[]) {
  const [saved, setSaved] = useState<string[] | null>(() => load())

  // ids que não existem mais no backend não são exibidos
  const selected = useMemo(() => (saved === null ? allIds : saved.filter((id) => allIds.includes(id))), [saved, allIds])

  useEffect(() => {
    if (saved) {
      try {
        sessionStorage.setItem(KEY, JSON.stringify(saved))
      } catch {
        /* sessionStorage indisponível: ignora */
      }
    }
  }, [saved])

  const toggle = useCallback(
    (id: string) => setSaved(selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]),
    [selected],
  )
  const selectAll = useCallback((ids: string[]) => setSaved(ids), [])
  const clear = useCallback(() => setSaved([]), [])
  return { selected, toggle, selectAll, clear }
}
