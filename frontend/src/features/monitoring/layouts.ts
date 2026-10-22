import type { LayoutId } from '@/types/api'

export const LAYOUTS: { id: LayoutId; label: string }[] = [
  { id: '1', label: '1' },
  { id: '2', label: '2' },
  { id: '4', label: '4' },
  { id: '6', label: '6' },
  { id: '9', label: '9' },
  { id: 'auto', label: 'Auto' },
]

const FIXED_COLS: Record<string, number> = { '1': 1, '2': 2, '4': 2, '6': 3, '9': 3 }

/** Quantas câmeras o layout comporta (auto => todas). Excedentes não são exibidas, mas continuam selecionadas. */
export function capacity(layout: LayoutId, count: number): number {
  return layout === 'auto' ? count : Math.min(Number(layout), count)
}

/** Colunas do grid, limitadas pela largura disponível (px) para manter painéis legíveis. */
export function columnsFor(layout: LayoutId, visibleCount: number, widthPx: number, minPanelPx = 280): number {
  const fit = Math.max(1, Math.floor(widthPx / minPanelPx))
  const wanted = layout === 'auto' ? Math.ceil(Math.sqrt(Math.max(visibleCount, 1))) : FIXED_COLS[layout]
  return Math.max(1, Math.min(wanted, fit, Math.max(visibleCount, 1)))
}
