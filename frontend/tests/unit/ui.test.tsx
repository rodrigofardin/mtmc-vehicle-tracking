import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mergeOrder, swapInOrder, visibleInOrder } from '@/features/camera-selection/order'
import { bestGrid } from '@/features/monitoring/layouts'
import { DEFAULT_PREFS, LIMITS, PREFS_KEY, sanitizePrefs } from '@/features/monitoring/prefs'
import { labelFor } from '@/features/tracking/drawOverlay'
import { isTypingTarget } from '@/hooks/useShortcuts'
import { Splitter } from '@/components/layout/Splitter'
import type { CameraStatus, Detection } from '@/types/api'

const apiMock = vi.hoisted(() => ({ cameras: vi.fn(), health: vi.fn(), runtime: vi.fn() }))
vi.mock('@/services/api/client', () => ({ api: apiMock, ApiError: class ApiError extends Error {} }))
const acquired = vi.hoisted(() => ({ ids: [] as string[] }))
vi.mock('@/services/streaming/StreamManager', () => ({
  streamManager: {
    acquire: (id: string) => {
      acquired.ids.push(id)
      return { stream: { currentState: 'connecting', onState: () => () => {}, onPacket: () => () => {} }, release: () => {} }
    },
  },
}))
import App from '@/App'

const cam = (id: string, extra: Partial<CameraStatus> = {}): CameraStatus => ({
  id, state: 'running', error: null, width: 1920, height: 1080, fps: 10, offset_s: 0, frames_total: 10, frame: 1, loop: 0,
  processing_fps: 8, inference_ms: 100, frames_published: 1, frames_dropped: 0, mode: 'live', ...extra,
})
const mk = (n: number) => Array.from({ length: n }, (_, i) => cam(`c${String(i + 1).padStart(3, '0')}`))
const tileOrder = () => screen.getAllByTestId(/^tile-/).map((e) => e.getAttribute('data-testid')!.replace('tile-', ''))

beforeEach(() => {
  localStorage.clear()
  sessionStorage.clear()
  acquired.ids.length = 0
  apiMock.cameras.mockReset()
  apiMock.health.mockReset()
  apiMock.runtime.mockReset()
  apiMock.health.mockResolvedValue({ status: 'ok', mode: 'live', environment: 't', device: 'cuda:0', uptime_s: 90, cameras: 0, ws_clients: 2, notes: [], association: { status: 'ativa', linked_identities: 3, last_run_ms: 1800 } })
  apiMock.runtime.mockResolvedValue(null)
})

// ------------------------------------------------------------------ lógica pura
describe('ordem das câmeras', () => {
  it('mescla a ordem salva com as câmeras conhecidas (novas no fim, sumidas fora, sem duplicatas)', () => {
    expect(mergeOrder(['c3', 'c1', 'zz', 'c3'], ['c1', 'c2', 'c3'])).toEqual(['c3', 'c1', 'c2'])
    expect(mergeOrder([], ['a', 'b'])).toEqual(['a', 'b'])
  })
  it('o grid mostra ordem ∩ selecionadas: selecionar/desmarcar não reordena', () => {
    const order = ['c3', 'c1', 'c2', 'c4']
    expect(visibleInOrder(order, ['c1', 'c2', 'c3'])).toEqual(['c3', 'c1', 'c2'])
    expect(visibleInOrder(order, ['c4', 'c1'])).toEqual(['c1', 'c4'])
  })
  it('troca duas câmeras (não "insere"): as demais permanecem onde estão', () => {
    expect(swapInOrder(['a', 'b', 'c', 'd'], 'a', 'c')).toEqual(['c', 'b', 'a', 'd'])
    expect(swapInOrder(['a', 'b'], 'a', 'a')).toEqual(['a', 'b'])
    expect(swapInOrder(['a', 'b'], 'a', 'x')).toEqual(['a', 'b'])
  })
})

describe('bestGrid (grade que maximiza a área de vídeo)', () => {
  it('uma câmera = 1×1; tela ultrawide com 5 câmeras usa uma linha', () => {
    expect(bestGrid(1, 1000, 600)).toEqual({ cols: 1, rows: 1 })
    expect(bestGrid(5, 3400, 600)).toEqual({ cols: 5, rows: 1 })
  })
  it('tela 16:9 típica com 4 câmeras = 2×2; tela alta com 2 câmeras empilha', () => {
    expect(bestGrid(4, 1600, 900)).toEqual({ cols: 2, rows: 2 })
    expect(bestGrid(2, 700, 1200)).toEqual({ cols: 1, rows: 2 })
  })
  it('nunca devolve colunas × linhas que não cabem todas as câmeras', () => {
    for (const n of [1, 2, 3, 5, 6, 7, 9, 12, 16]) for (const [w, h] of [[800, 500], [1920, 1000], [3440, 900], [500, 900]]) {
      const g = bestGrid(n, w, h)
      expect(g.cols * g.rows).toBeGreaterThanOrEqual(n)
    }
  })
})

describe('preferências persistidas', () => {
  it('valida e completa dados corrompidos ou de versão antiga', () => {
    const p = sanitizePrefs({ layout: 'xx', sidebar: { collapsed: 'sim', width: 99999 }, panel: { open: true, width: 10, tab: 'nope' }, overlay: { boxes: false }, order: ['a', 3, 'b'], overlayOverrides: { c1: { trails: false, foo: 1 }, c2: {} } })
    expect(p.layout).toBe(DEFAULT_PREFS.layout)
    expect(p.sidebar.collapsed).toBe(false)
    expect(p.sidebar.width).toBe(LIMITS.sidebar.max)
    expect(p.panel.width).toBe(LIMITS.panel.min)
    expect(p.panel.tab).toBe('overview')
    expect(p.panel.open).toBe(true)
    expect(p.overlay).toEqual({ boxes: false, ids: true, labels: true, trails: true })
    expect(p.order).toEqual(['a', 'b'])
    expect(p.overlayOverrides).toEqual({ c1: { trails: false } })
  })
  it('entrada inválida cai nos padrões', () => {
    expect(sanitizePrefs(null)).toEqual(DEFAULT_PREFS)
    expect(sanitizePrefs('lixo')).toEqual(DEFAULT_PREFS)
  })
})

describe('rótulos e atalhos', () => {
  const d: Detection = { id: 12, cls: 2, name: 'car', conf: 0.87, bbox: [0, 0, 1, 1], gid: 'G0004', uncertain: true }
  it('IDs e classe/confiança são independentes', () => {
    expect(labelFor(d)).toBe('#12 car 87% G0004 ?')
    expect(labelFor(d, { ids: true, labels: false })).toBe('#12 G0004 ?')
    expect(labelFor(d, { ids: false, labels: true })).toBe('car 87% ?')
    expect(labelFor(d, { ids: false, labels: false })).toBe('')
  })
  it('atalhos não disparam ao digitar em campos', () => {
    const i = document.createElement('input')
    const cb = document.createElement('input')
    cb.type = 'checkbox'
    expect(isTypingTarget(i)).toBe(true)
    expect(isTypingTarget(document.createElement('textarea'))).toBe(true)
    expect(isTypingTarget(document.createElement('select'))).toBe(true)
    expect(isTypingTarget(cb)).toBe(false)
    expect(isTypingTarget(document.createElement('div'))).toBe(false)
  })
})

describe('Splitter', () => {
  it('teclado: setas, Shift, Home/End e Enter (padrão)', () => {
    const on = vi.fn()
    const reset = vi.fn()
    render(<Splitter mode="grow-right" width={300} min={200} max={420} onResize={on} onReset={reset} label="Teste" />)
    const s = screen.getByRole('separator', { name: 'Teste' })
    expect(s).toHaveAttribute('aria-valuenow', '300')
    fireEvent.keyDown(s, { key: 'ArrowRight' }); expect(on).toHaveBeenLastCalledWith(316)
    fireEvent.keyDown(s, { key: 'ArrowLeft', shiftKey: true }); expect(on).toHaveBeenLastCalledWith(236)
    fireEvent.keyDown(s, { key: 'End' }); expect(on).toHaveBeenLastCalledWith(420)
    fireEvent.keyDown(s, { key: 'Home' }); expect(on).toHaveBeenLastCalledWith(200)
    fireEvent.keyDown(s, { key: 'Enter' }); expect(reset).toHaveBeenCalled()
  })
  it('o painel da direita cresce arrastando para a esquerda (e respeita os limites)', () => {
    const on = vi.fn()
    render(<Splitter mode="grow-left" width={500} min={380} max={600} onResize={on} label="Dir" />)
    const s = screen.getByRole('separator', { name: 'Dir' })
    fireEvent.keyDown(s, { key: 'ArrowLeft', shiftKey: true }); expect(on).toHaveBeenLastCalledWith(564)
    fireEvent.keyDown(s, { key: 'ArrowLeft', shiftKey: true }); expect(on).toHaveBeenLastCalledWith(564)
  })
})

// ------------------------------------------------------------------ integração na página
describe('página: ordem, barras, controles e atalhos', () => {
  it('restaura a ordem salva e a mantém ao desmarcar/selecionar; câmera nova entra no fim', async () => {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ order: ['c003', 'c001', 'c002'] }))
    apiMock.cameras.mockResolvedValue(mk(4))
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^tile-/)).toHaveLength(4))
    expect(tileOrder()).toEqual(['c003', 'c001', 'c002', 'c004'])
    await u.click(screen.getByRole('checkbox', { name: 'Selecionar c001' }))
    expect(tileOrder()).toEqual(['c003', 'c002', 'c004'])
    await u.click(screen.getByRole('checkbox', { name: 'Selecionar c001' }))
    expect(tileOrder()).toEqual(['c003', 'c001', 'c002', 'c004']) // volta à posição preferida
  })

  it('lista de câmeras: recolhe para o modo compacto, preserva a seleção e expande de volta', async () => {
    apiMock.cameras.mockResolvedValue(mk(3))
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^tile-/)).toHaveLength(3))
    await u.click(screen.getByRole('checkbox', { name: 'Selecionar c002' }))
    await u.click(screen.getByTestId('sidebar-toggle'))
    expect(screen.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'true')
    expect(screen.getByTestId('rail-cam-c001')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('rail-cam-c002')).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getAllByTestId(/^tile-/)).toHaveLength(2) // grade não foi afetada
    await u.click(screen.getByTestId('rail-cam-c002')) // seleção também funciona no modo compacto
    expect(screen.getAllByTestId(/^tile-/)).toHaveLength(3)
    await u.click(screen.getByTestId('sidebar-toggle'))
    expect(screen.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'false')
    expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}').sidebar).toBeDefined()
  })

  it('preferências persistem entre sessões (recolhimento, layout e aba do painel)', async () => {
    localStorage.setItem(PREFS_KEY, JSON.stringify({ layout: '4', sidebar: { collapsed: true, width: 300 }, panel: { open: false, width: 700, tab: 'params' } }))
    apiMock.cameras.mockResolvedValue(mk(6))
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '4')) // layout 4 restaurado
    expect(screen.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'true')
    expect(screen.getByTestId('layout-4')).toHaveAttribute('aria-checked', 'true')
  })

  it('atalhos: layout, lista (B), painel (P), ajuda (?) e suspensão ao digitar', async () => {
    apiMock.cameras.mockResolvedValue(mk(6))
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^tile-/)).toHaveLength(6))
    await u.keyboard('4')
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '4')
    await u.keyboard('a')
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '6')
    await u.keyboard('b')
    expect(screen.getByTestId('sidebar')).toHaveAttribute('data-collapsed', 'true')
    await u.keyboard('p')
    expect(screen.getByTestId('processing-panel')).toHaveAttribute('data-open', 'true')
    await u.keyboard('p')
    expect(screen.getByTestId('processing-panel')).toHaveAttribute('data-open', 'false')
    await u.keyboard('?')
    expect(await screen.findByTestId('help-popover')).toHaveTextContent('Reordenar')

    await u.keyboard('{Escape}')
    await u.keyboard('b') // expande de novo
    const search = screen.getByLabelText('Buscar câmera')
    await u.click(search)
    await u.keyboard('4') // digitando no campo: NÃO muda o layout
    expect(search).toHaveValue('4')
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '6')
  })

  it('"Limpar" pede confirmação quando há mais de uma câmera; cancelar não remove nada', async () => {
    apiMock.cameras.mockResolvedValue(mk(3))
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^tile-/)).toHaveLength(3))
    await u.click(screen.getByRole('button', { name: 'Limpar' }))
    const dlg = await screen.findByRole('alertdialog')
    expect(dlg).toHaveTextContent('processamento no backend')
    await u.click(within(dlg).getByRole('button', { name: 'Cancelar' }))
    expect(screen.getAllByTestId(/^tile-/)).toHaveLength(3)
  })

  it('controles por câmera: alternam só aquela câmera; "todas" no cabeçalho substitui a sobrescrita', async () => {
    apiMock.cameras.mockResolvedValue(mk(2))
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^tile-/)).toHaveLength(2))
    const t1 = screen.getByTestId('tgl-trails-c001')
    const t2 = screen.getByTestId('tgl-trails-c002')
    expect(t1).toHaveAttribute('aria-pressed', 'true')
    await u.click(t1)
    expect(t1).toHaveAttribute('aria-pressed', 'false')
    expect(t2).toHaveAttribute('aria-pressed', 'true') // a outra câmera não mudou
    await waitFor(() => expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}').overlayOverrides?.c001).toEqual({ trails: false }))
    await u.click(screen.getByTestId('global-trails')) // desliga em todas e limpa a sobrescrita
    expect(screen.getByTestId('tgl-trails-c001')).toHaveAttribute('aria-pressed', 'false')
    expect(screen.getByTestId('tgl-trails-c002')).toHaveAttribute('aria-pressed', 'false')
    await waitFor(() => expect(JSON.parse(localStorage.getItem(PREFS_KEY) ?? '{}').overlayOverrides).toEqual({}))
  })

  it('maximizar/restaurar e alternar layouts não reabre conexões nem remonta os painéis', async () => {
    apiMock.cameras.mockResolvedValue(mk(3))
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^tile-/)).toHaveLength(3))
    const inst = () => screen.getAllByTestId(/^panel-/).map((e) => e.getAttribute('data-instance'))
    const before = inst()
    const acquiredBefore = acquired.ids.length
    await u.click(screen.getByTestId('expand-c002'))
    await u.click(screen.getByTestId('expand-c002'))
    await u.keyboard('2')
    await u.keyboard('a')
    expect(inst()).toEqual(before) // mesmas instâncias: nenhum painel foi remontado
    expect(acquired.ids.length).toBe(acquiredBefore) // nenhuma nova aquisição de stream
  })

  it('barra superior: contadores de câmeras conectadas, com falha e selecionadas', async () => {
    apiMock.cameras.mockResolvedValue([cam('c001'), cam('c002', { state: 'error', error: 'x' }), cam('c003', { state: 'unavailable', error: 'y' }), cam('c004', { state: 'stalled' })])
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('count-total')).toHaveTextContent('4'))
    expect(screen.getByTestId('count-processing')).toHaveTextContent('1')
    expect(screen.getByTestId('count-failed')).toHaveTextContent('3')
    expect(screen.getByTestId('metric-perf')).toHaveTextContent('8.0')
  })
})
