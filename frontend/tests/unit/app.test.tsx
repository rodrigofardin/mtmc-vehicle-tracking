import { act, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { CameraStatus, FramePacket, StreamState } from '@/types/api'

// ---- fakes: API e StreamManager (nenhuma rede real) ----
const apiMock = vi.hoisted(() => ({ cameras: vi.fn(), health: vi.fn() }))
vi.mock('@/services/api/client', () => ({
  api: apiMock,
  ApiError: class ApiError extends Error {},
}))

class FakeStream {
  state: StreamState = 'streaming'
  attempt = 0
  pl = new Set<(p: FramePacket) => void>()
  sl = new Set<(s: StreamState, i: { attempt: number }) => void>()
  readonly camId: string
  constructor(camId: string) {
    this.camId = camId
  }
  get currentState() { return this.state }
  onPacket(f: (p: FramePacket) => void) { this.pl.add(f); return () => this.pl.delete(f) }
  onState(f: (s: StreamState, i: { attempt: number }) => void) { this.sl.add(f); f(this.state, { attempt: this.attempt }); return () => this.sl.delete(f) }
  emit(p: FramePacket) { this.pl.forEach((f) => f(p)) }
  setState(s: StreamState, attempt = 0) { this.state = s; this.attempt = attempt; this.sl.forEach((f) => f(s, { attempt })) }
}
const hub = vi.hoisted(() => ({ streams: new Map<string, unknown>(), acquired: [] as string[] }))
vi.mock('@/services/streaming/StreamManager', () => ({
  streamManager: {
    acquire: (id: string) => {
      hub.acquired.push(id)
      let s = hub.streams.get(id) as FakeStream | undefined
      if (!s) { s = new FakeStream(id); hub.streams.set(id, s) }
      return { stream: s, release: () => {} }
    },
  },
}))

import App from '@/App'

const cam = (id: string, extra: Partial<CameraStatus> = {}): CameraStatus => ({
  id, state: 'running', error: null, width: 1920, height: 1080, fps: 10, offset_s: 0, frames_total: 100,
  frame: 5, loop: 0, processing_fps: 9.5, inference_ms: 20, frames_published: 10, frames_dropped: 0, mode: 'live', ...extra,
})
const health = { status: 'ok', mode: 'live', environment: 'teste', device: 'cuda:0', uptime_s: 1, cameras: 0, ws_clients: 0, notes: [] }
const mkCams = (n: number) => Array.from({ length: n }, (_, i) => cam(`c${String(i + 1).padStart(3, '0')}`))
const stream = (id: string) => hub.streams.get(id) as FakeStream

function packet(camId: string, dets: FramePacket['header']['detections'], frame = 1): FramePacket {
  return {
    jpeg: new Blob([new Uint8Array([0xff, 0xd8])]),
    header: { cam: camId, seq: frame, frame, loop: 0, t_global: 12.3, t_local: 12.3, width: 1920, height: 1080, stream_width: 960, stream_height: 540, proc_fps: 10, inference_ms: 21, mode: 'live', detections: dets },
  }
}

beforeEach(() => {
  sessionStorage.clear()
  localStorage.clear()
  hub.streams.clear()
  hub.acquired.length = 0
  apiMock.cameras.mockReset()
  apiMock.health.mockReset()
  apiMock.health.mockResolvedValue(health)
})

describe('App', () => {
  it('mostra carregando e depois descobre câmeras dinamicamente', async () => {
    apiMock.cameras.mockResolvedValue(mkCams(3))
    render(<App />)
    expect(screen.getByText('Carregando câmeras…')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('count-total')).toHaveTextContent('3'))
    expect(screen.getByTestId('count-selected')).toHaveTextContent('3')
    expect(screen.getByTestId('backend-status')).toHaveAttribute('data-connection', 'online')
    expect(screen.getAllByTestId(/^panel-/)).toHaveLength(3)
  })

  it.each([1, 2, 5, 7, 12])('suporta %i câmeras (layout auto)', async (n) => {
    apiMock.cameras.mockResolvedValue(mkCams(n))
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^panel-/)).toHaveLength(n))
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', String(n))
  })

  it('backend indisponível: mensagem clara e nenhuma câmera inventada', async () => {
    apiMock.cameras.mockRejectedValue(new Error('Backend inacessível (/cameras)'))
    render(<App />)
    await waitFor(() => expect(within(screen.getByRole('main')).getByText('Backend indisponível')).toBeInTheDocument())
    expect(screen.getByTestId('backend-status')).toHaveAttribute('data-connection', 'offline')
    expect(screen.queryAllByTestId(/^panel-/)).toHaveLength(0)
    expect(screen.getByRole('button', { name: 'Tentar novamente' })).toBeInTheDocument()
  })

  it('seleção: desmarcar remove do grid; limpar e selecionar todas; busca', async () => {
    const u = userEvent.setup()
    apiMock.cameras.mockResolvedValue(mkCams(4))
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^panel-/)).toHaveLength(4))

    await u.click(screen.getByRole('checkbox', { name: 'Selecionar c002' }))
    expect(screen.queryByTestId('panel-c002')).not.toBeInTheDocument()
    expect(screen.getByTestId('count-selected')).toHaveTextContent('3')

    await u.click(screen.getByRole('button', { name: 'Limpar' }))
    // com mais de uma câmera selecionada, limpar pede confirmação (e nada some antes de confirmar)
    expect(screen.getAllByTestId(/^panel-/)).toHaveLength(3)
    await u.click(screen.getByTestId('confirm-clear'))
    expect(screen.getByText('Nenhuma câmera selecionada')).toBeInTheDocument()

    await u.click(screen.getByRole('button', { name: 'Selecionar todas' }))
    expect(screen.getAllByTestId(/^panel-/)).toHaveLength(4)

    await u.type(screen.getByLabelText('Buscar câmera'), 'c003')
    const list = screen.getByTestId('camera-list')
    expect(within(list).getAllByRole('checkbox')).toHaveLength(1)
    expect(sessionStorage.getItem('mtmc.selection.v1')).toContain('c003')
  })

  it('remover pelo painel e alterar o layout limita o número de painéis', async () => {
    const u = userEvent.setup()
    apiMock.cameras.mockResolvedValue(mkCams(6))
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^panel-/)).toHaveLength(6))
    await u.click(screen.getByTestId('layout-4'))
    // o layout limita o que é EXIBIDO; os demais painéis seguem montados (ocultos) para não reiniciar streams
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '4')
    expect(screen.getByRole('note')).toHaveTextContent('Exibindo 4 de 6')
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-cols', '2')
    await u.click(screen.getByTestId('layout-1'))
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '1')
    await u.click(screen.getByTestId('layout-auto'))
    await u.click(within(screen.getByTestId('panel-c001')).getByRole('button', { name: /Remover c001/ }))
    expect(screen.queryByTestId('panel-c001')).not.toBeInTheDocument()
  })

  it('ampliar uma câmera e voltar ao grid (Esc)', async () => {
    const u = userEvent.setup()
    apiMock.cameras.mockResolvedValue(mkCams(3))
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^panel-/)).toHaveLength(3))
    await u.click(within(screen.getByTestId('panel-c002')).getByRole('button', { name: /Maximizar c002/ }))
    // maximizar só muda a visualização: os outros painéis continuam montados (conexão mantida), apenas ocultos
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '1')
    expect(screen.getAllByTestId(/^panel-/)).toHaveLength(3)
    expect(screen.getByTestId('tile-c001')).toHaveClass('hidden')
    expect(screen.getByTestId('tile-c002')).not.toHaveClass('hidden')
    await u.keyboard('{Escape}')
    expect(screen.getByTestId('video-grid')).toHaveAttribute('data-count', '3')
    expect(screen.getByTestId('tile-c001')).not.toHaveClass('hidden')
  })

  it('não reabre conexão ao mudar de layout (uma aquisição por câmera montada)', async () => {
    const u = userEvent.setup()
    apiMock.cameras.mockResolvedValue(mkCams(2))
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^panel-/)).toHaveLength(2))
    const before = hub.streams.size
    await u.click(screen.getByTestId('layout-2'))
    await u.click(screen.getByTestId('layout-auto'))
    expect(hub.streams.size).toBe(before) // mesmas instâncias reaproveitadas pelo manager
  })

  it('exibe metadados do fluxo e diferencia "sem detecções" de falha', async () => {
    apiMock.cameras.mockResolvedValue(mkCams(1))
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('panel-c001')).toBeInTheDocument())
    await waitFor(() => expect(hub.streams.has('c001')).toBe(true))
    const s = stream('c001')
    await act(async () => {
      s.emit(packet('c001', [{ id: 4, cls: 2, name: 'car', conf: 0.9, bbox: [10, 10, 80, 60], gid: null, uncertain: false }]))
      await new Promise((r) => setTimeout(r, 450))
    })
    const stats = screen.getByTestId('stats-c001')
    expect(stats).toHaveTextContent('1 obj')
    expect(stats).toHaveTextContent('21 ms')
    expect(stats).toHaveTextContent('t 12.3 s')
    expect(screen.queryByTestId('no-detections')).not.toBeInTheDocument()

    await act(async () => {
      s.emit(packet('c001', [], 2))
      await new Promise((r) => setTimeout(r, 450))
    })
    expect(screen.getByTestId('no-detections')).toHaveTextContent('Sem detecções neste frame')
    expect(screen.getByTestId('panel-c001').querySelector('[data-health]')).toHaveAttribute('data-health', 'live')
  })

  it('estados de erro: reconectando, vídeo indisponível e backend com erro', async () => {
    apiMock.cameras.mockResolvedValue([cam('c001'), cam('c002', { state: 'unavailable', error: 'arquivo inexistente' }), cam('c003', { state: 'error', error: 'falha de leitura' })])
    render(<App />)
    await waitFor(() => expect(screen.getAllByTestId(/^panel-/)).toHaveLength(3))
    expect(within(screen.getByTestId('panel-c002')).getByText('Vídeo indisponível')).toBeInTheDocument()
    expect(within(screen.getByTestId('panel-c003')).getByText('Erro na câmera')).toBeInTheDocument()
    await waitFor(() => expect(hub.streams.has('c001')).toBe(true))
    await act(async () => stream('c001').setState('reconnecting', 2))
    await waitFor(() => expect(within(screen.getByTestId('panel-c001')).getByText('Reconectando…')).toBeInTheDocument(), { timeout: 2000 })
    // câmera indisponível nunca abre fluxo
    expect(hub.acquired).not.toContain('c002')
  })

  it('não declara "ao vivo" sem confirmação do backend', async () => {
    apiMock.cameras.mockResolvedValue([cam('c001', { state: 'starting' })])
    render(<App />)
    await waitFor(() => expect(hub.streams.has('c001')).toBe(true))
    await act(async () => stream('c001').emit(packet('c001', [])))
    expect(screen.getByTestId('panel-c001').querySelector('[data-health]')).not.toHaveAttribute('data-health', 'live')
  })
})
