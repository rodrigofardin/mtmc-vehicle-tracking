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

export interface GridSpec {
  cols: number
  rows: number
}

/**
 * Escolhe colunas × linhas que MAXIMIZAM a área de vídeo (proporção `aspect`) no espaço disponível, evitando faixas pretas e
 * grandes vazios (ex.: 5 câmeras em tela ultrawide vira 5×1 em vez de 3×2). Desempata por menos células vazias e menos colunas.
 * `chrome` = altura fixa por tile (rodapé) descontada da altura útil do vídeo.
 */
export function bestGrid(n: number, width: number, height: number, aspect = 16 / 9, gap = 6, chrome = 22): GridSpec {
  if (n <= 1) return { cols: 1, rows: 1 }
  let best: GridSpec & { area: number; empty: number } = { cols: 1, rows: n, area: -1, empty: 0 }
  for (let cols = 1; cols <= n; cols++) {
    const rows = Math.ceil(n / cols)
    const cw = (width - gap * (cols - 1)) / cols
    const ch = (height - gap * (rows - 1)) / rows - chrome
    if (cw <= 0 || ch <= 0) continue
    const vw = Math.min(cw, ch * aspect)
    const area = vw * (vw / aspect)
    const empty = cols * rows - n
    const better = area > best.area * 1.0001 || (Math.abs(area - best.area) <= best.area * 1e-4 && (empty < best.empty || (empty === best.empty && cols < best.cols)))
    if (better) best = { cols, rows, area, empty }
  }
  return { cols: best.cols, rows: best.rows }
}
