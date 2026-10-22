import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import { readJson, writeJson } from '@/lib/storage'
import type { LayoutId } from '@/types/api'

export type PanelTab = 'overview' | 'models' | 'params' | 'cameras' | 'history'

export interface UiPrefs {
  layout: LayoutId
  sidebar: { collapsed: boolean; width: number }
  panel: { open: boolean; width: number; tab: PanelTab }
  overlay: OverlayOptions // padrão global (aplicado a todas as câmeras)
  overlayOverrides: Record<string, Partial<OverlayOptions>> // sobrescritas por câmera
  order: string[] // ordem preferida das câmeras (todas as conhecidas)
}

export const PREFS_KEY = 'mtmc.ui.v1'
export const LIMITS = {
  sidebar: { min: 200, max: 420, compact: 56, def: 248 },
  panel: { min: 380, max: 980, def: 640 },
}

export const DEFAULT_PREFS: UiPrefs = {
  layout: 'auto',
  sidebar: { collapsed: false, width: LIMITS.sidebar.def },
  panel: { open: false, width: LIMITS.panel.def, tab: 'overview' },
  overlay: { boxes: true, ids: true, labels: true, trails: true },
  overlayOverrides: {},
  order: [],
}

const LAYOUTS: LayoutId[] = ['1', '2', '4', '6', '9', 'auto']
const TABS: PanelTab[] = ['overview', 'models', 'params', 'cameras', 'history']
const clamp = (v: unknown, lo: number, hi: number, d: number) => (typeof v === 'number' && Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : d)
const bool = (v: unknown, d: boolean) => (typeof v === 'boolean' ? v : d)

/** Valida o que veio do armazenamento (pode estar corrompido ou de versão antiga) e completa com os padrões. */
export function sanitizePrefs(raw: unknown): UiPrefs {
  const r = (raw && typeof raw === 'object' ? raw : {}) as Record<string, any> // eslint-disable-line @typescript-eslint/no-explicit-any
  const d = DEFAULT_PREFS
  const ov = (o: any): OverlayOptions => ({ // eslint-disable-line @typescript-eslint/no-explicit-any
    boxes: bool(o?.boxes, d.overlay.boxes), ids: bool(o?.ids, d.overlay.ids), labels: bool(o?.labels, d.overlay.labels), trails: bool(o?.trails, d.overlay.trails),
  })
  const overrides: Record<string, Partial<OverlayOptions>> = {}
  if (r.overlayOverrides && typeof r.overlayOverrides === 'object') {
    for (const [cam, v] of Object.entries(r.overlayOverrides as Record<string, any>)) { // eslint-disable-line @typescript-eslint/no-explicit-any
      const o: Partial<OverlayOptions> = {}
      for (const k of ['boxes', 'ids', 'labels', 'trails'] as const) if (typeof v?.[k] === 'boolean') o[k] = v[k]
      if (Object.keys(o).length) overrides[cam] = o
    }
  }
  return {
    layout: LAYOUTS.includes(r.layout) ? r.layout : d.layout,
    sidebar: { collapsed: bool(r.sidebar?.collapsed, false), width: clamp(r.sidebar?.width, LIMITS.sidebar.min, LIMITS.sidebar.max, d.sidebar.width) },
    panel: {
      open: bool(r.panel?.open, false),
      width: clamp(r.panel?.width, LIMITS.panel.min, LIMITS.panel.max, d.panel.width),
      tab: TABS.includes(r.panel?.tab) ? r.panel.tab : d.panel.tab,
    },
    overlay: ov(r.overlay),
    overlayOverrides: overrides,
    order: Array.isArray(r.order) ? r.order.filter((x: unknown): x is string => typeof x === 'string') : [],
  }
}

export const loadPrefs = (): UiPrefs => sanitizePrefs(readJson<unknown>(PREFS_KEY, null))
export const savePrefs = (p: UiPrefs) => writeJson(PREFS_KEY, p)
export type { LayoutId }
