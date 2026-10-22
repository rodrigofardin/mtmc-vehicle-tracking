/**
 * Ordem das câmeras no grid.
 *  - `order` é a ordem preferida de TODAS as câmeras conhecidas (persistida).
 *  - Câmeras novas entram no fim; câmeras que sumiram do backend saem da ordem efetiva (mas ficam salvas).
 *  - O grid mostra `order ∩ selecionadas`: selecionar/desmarcar nunca reordena as demais.
 */
export function mergeOrder(saved: string[], ids: string[]): string[] {
  const known = new Set(ids)
  const kept = saved.filter((id, i) => known.has(id) && saved.indexOf(id) === i)
  const keptSet = new Set(kept)
  return [...kept, ...ids.filter((id) => !keptSet.has(id))]
}

export function visibleInOrder(order: string[], selected: string[]): string[] {
  const sel = new Set(selected)
  return order.filter((id) => sel.has(id))
}

/** Troca a posição de duas câmeras na ordem completa. Ids desconhecidos: devolve a ordem original. */
export function swapInOrder(order: string[], a: string, b: string): string[] {
  const i = order.indexOf(a)
  const j = order.indexOf(b)
  if (i < 0 || j < 0 || i === j) return order
  const out = order.slice()
  out[i] = b
  out[j] = a
  return out
}
