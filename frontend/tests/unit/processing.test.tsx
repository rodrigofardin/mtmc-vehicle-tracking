import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { RuntimeInfo, RuntimeParam } from '@/types/api'

const apiMock = vi.hoisted(() => ({ cameras: vi.fn(), health: vi.fn(), runtime: vi.fn() }))
vi.mock('@/services/api/client', () => ({ api: apiMock, ApiError: class ApiError extends Error {} }))
vi.mock('@/services/streaming/StreamManager', () => ({
  streamManager: { acquire: () => ({ stream: { currentState: 'connecting', onState: () => () => {}, onPacket: () => () => {} }, release: () => {} }) },
}))

import App from '@/App'
import { TooltipProvider } from '@/components/ui/tooltip'
import { ProcessingPanel as RawPanel } from '@/features/processing/ProcessingPanel'

// Tooltip exige o provider (no app ele vem de App.tsx)
const ProcessingPanel = (props: React.ComponentProps<typeof RawPanel>) => (
  <TooltipProvider>
    <RawPanel {...props} />
  </TooltipProvider>
)

const P = (id: string, extra: Partial<RuntimeParam> = {}): RuntimeParam => ({
  id, label: id, category: 'Inferência', scope: 'modelo', configured: 0.25, applied: 0.25, status: 'confirmed',
  source: 'runtime', description: `descrição de ${id}`, unit: null, editable: false, fixed_in_code: false, ...extra,
})

const rt = (over: Partial<RuntimeInfo> = {}): RuntimeInfo => ({
  schema_version: 1, read_only: true, generated_at: '2026-10-22T10:00:00', mode: 'live', started_at: '2026-10-22T09:00:00',
  revision: 'abc123def456',
  config: { file: '/workspace/configs/live.yaml', file_mtime: '2026-10-22T08:00:00', sha256: '0123456789abcdef', changed_on_disk: false, diff: [] },
  versions: { python: '3.11.13', torch: '2.7.1+cu126', tensorrt: '10.16.1.11', gpu: 'NVIDIA GeForce RTX 2060', cuda: '12.6' },
  history: { supported: false, reason: 'o backend não registra alterações (somente leitura)' },
  models: [
    { id: 'detector', role: 'detector', name: 'yolov8l-trt', artifact: '/workspace/models/yolov8l.pt', task: 'detect', format: 'TensorRT (engine)',
      backend_class: 'TensorRTBackend', device: 'cuda:0', load_state: 'carregado', input_shape: [1, 3, 544, 960], io_dtype: 'FLOAT',
      engine_metadata: { imgsz: [544, 960], batch: 1, args: { half: true }, version: 'mtmc-trt' },
      classes: [{ id: 2, name: 'car' }, { id: 7, name: 'truck' }], n_classes: 2, cameras: ['c001', 'c002'] },
    { id: 'tracker', role: 'tracker', name: 'BYTETracker', artifact: null, format: 'algoritmo (sem pesos)', load_state: 'ativo', classes: [], n_classes: null, cameras: ['c001', 'c002'] },
  ],
  parameters: [
    P('detector.conf'),
    P('detector.iou', { configured: 0.7, applied: 0.7, fixed_in_code: true }),
    P('tracker.track_buffer', { category: 'Tracking', scope: 'câmera', configured: 30, applied: 30 }),
    P('association.accept_score', { category: 'Associação', scope: 'global', configured: 0.8, applied: 0.5, status: 'diverges' }),
    P('spatial.zones', { category: 'Regras e regiões', scope: 'câmera', configured: {}, applied: null, status: 'unconfirmed' }),
  ],
  cameras: [
    { id: 'c001', state: 'running', error: null, model: 'yolov8l-trt', model_load: 'carregado', tracker: 'BYTETracker',
      params: { 'detector.conf': { value: 0.25, origin: 'global' }, 'time.offset_s': { value: 0, origin: 'metadados (cam_timestamp)' } } },
    { id: 'c002', state: 'running', error: null, model: 'yolov8l-trt', model_load: 'carregado', tracker: 'BYTETracker',
      params: { 'detector.conf': { value: 0.4, origin: 'camera' }, 'time.offset_s': { value: 1.64, origin: 'metadados (cam_timestamp)' } } },
  ],
  association: {},
  ...over,
})

const runtimeData = (data: RuntimeInfo | null, error: string | null = null) => ({ data, loading: false, error, refresh: vi.fn() })

describe('ProcessingPanel (somente leitura)', () => {
  it('visão geral: origem, revisão, versões e contagem de estados', () => {
    render(<ProcessingPanel runtime={runtimeData(rt())} onClose={() => {}} />)
    const t = screen.getByTestId('tab-overview')
    expect(t).toHaveTextContent('/workspace/configs/live.yaml')
    expect(t).toHaveTextContent('abc123def456')
    expect(t).toHaveTextContent('2.7.1+cu126')
    expect(t).toHaveTextContent('NVIDIA GeForce RTX 2060')
    expect(screen.getByText('somente leitura')).toBeInTheDocument()
    expect(screen.queryByTestId('config-changed')).not.toBeInTheDocument()
  })

  it('avisa quando o arquivo mudou depois da inicialização e mostra valor rodando × disco', () => {
    const r = rt({ config: { ...rt().config, changed_on_disk: true, diff: [{ key: 'association.accept_score', running: 0.5, on_disk: 0.8 }] } })
    render(<ProcessingPanel runtime={runtimeData(r)} onClose={() => {}} />)
    const a = screen.getByTestId('config-changed')
    expect(a).toHaveTextContent('association.accept_score')
    expect(a).toHaveTextContent('0.5')
    expect(a).toHaveTextContent('0.8')
    expect(a).toHaveTextContent('somente leitura')
  })

  it('modelos: metadados reais, classes e "não disponível" onde o runtime não informa', async () => {
    const u = userEvent.setup()
    render(<ProcessingPanel runtime={runtimeData(rt())} onClose={() => {}} />)
    await u.click(screen.getByRole('tab', { name: 'Modelos' }))
    await u.click(screen.getByRole('button', { name: /yolov8l-trt/ }))
    const m = screen.getByTestId('model-detector')
    expect(m).toHaveTextContent('TensorRT (engine)')
    expect(m).toHaveTextContent('[1, 3, 544, 960]')
    expect(m).toHaveTextContent('cuda:0')
    expect(m).toHaveTextContent('FP16')
    expect(m).toHaveTextContent('não confirmado pelo runtime') // o FP16 é metadado do engine, não do runtime
    expect(m).toHaveTextContent('não disponível (os pesos não trazem versão')
    expect(within(screen.getByTestId('class-list')).getAllByText(/·/)).toHaveLength(2)
    expect(m).toHaveTextContent('c001, c002')
  })

  it('modo replay: informa que nenhum modelo executa', async () => {
    const u = userEvent.setup()
    render(<ProcessingPanel runtime={runtimeData(rt({ mode: 'replay', models: [] }))} onClose={() => {}} />)
    await u.click(screen.getByRole('tab', { name: 'Modelos' }))
    expect(screen.getByTestId('tab-models')).toHaveTextContent('Nenhum modelo em execução')
  })

  it('parâmetros: não confirmado, divergente, fixo no código e filtros', async () => {
    const u = userEvent.setup()
    render(<ProcessingPanel runtime={runtimeData(rt())} onClose={() => {}} />)
    await u.click(screen.getByRole('tab', { name: 'Limiares e parâmetros' }))
    expect(screen.getByTestId('param-count')).toHaveTextContent('5 de 5')
    expect(within(screen.getByTestId('param-spatial.zones')).getByText('não confirmado pelo runtime')).toBeInTheDocument()
    const div = screen.getByTestId('param-association.accept_score')
    expect(div).toHaveTextContent('0.8')
    expect(div).toHaveTextContent('0.5')
    expect(div.querySelector('[data-status="diverges"]')).not.toBeNull()
    expect(within(screen.getByTestId('param-detector.iou')).getByText('fixo no código')).toBeInTheDocument()

    await u.selectOptions(screen.getByLabelText('estado'), 'diverges')
    expect(screen.getByTestId('param-count')).toHaveTextContent('1 de 5')
    await u.selectOptions(screen.getByLabelText('estado'), 'todas')
    await u.selectOptions(screen.getByLabelText('categoria'), 'Tracking')
    expect(screen.getByTestId('param-count')).toHaveTextContent('1 de 5')
    await u.selectOptions(screen.getByLabelText('categoria'), 'todas')
    await u.type(screen.getByLabelText('Buscar parâmetro'), 'iou')
    expect(screen.getByTestId('param-count')).toHaveTextContent('1 de 5')
    await u.clear(screen.getByLabelText('Buscar parâmetro'))
    await u.type(screen.getByLabelText('Buscar parâmetro'), 'zzz')
    expect(screen.getByText('Nenhum parâmetro com esses filtros')).toBeInTheDocument()
  })

  it('por câmera: origem herdada × sobrescrita e destaque de divergências entre câmeras', async () => {
    const u = userEvent.setup()
    render(<ProcessingPanel runtime={runtimeData(rt())} onClose={() => {}} />)
    await u.click(screen.getByRole('tab', { name: 'Por câmera' }))
    const row = screen.getByTestId('camrow-detector.conf')
    expect(row).toHaveTextContent('0.25')
    expect(row).toHaveTextContent('0.4')
    expect(row).toHaveTextContent('herdado (global)')
    expect(row).toHaveTextContent('sobrescrito na câmera')
    expect(screen.getByTestId('cams-diff-note')).toHaveTextContent('há parâmetros que diferem')
    await u.click(screen.getByLabelText('Mostrar só divergências'))
    expect(screen.getByTestId('camrow-detector.conf')).toBeInTheDocument()
  })

  it('histórico: informa que não há suporte no backend', async () => {
    const u = userEvent.setup()
    render(<ProcessingPanel runtime={runtimeData(rt())} onClose={() => {}} />)
    await u.click(screen.getByRole('tab', { name: 'Histórico' }))
    expect(screen.getByTestId('tab-history')).toHaveTextContent('Sem histórico de alterações')
  })

  it('erro da API: sem dados mostra alerta; com dados anteriores mantém a última consulta', () => {
    const { rerender } = render(<ProcessingPanel runtime={runtimeData(null, 'Backend inacessível (/runtime)')} onClose={() => {}} />)
    expect(screen.getByRole('alert')).toHaveTextContent('Não foi possível consultar o backend')
    rerender(<ProcessingPanel runtime={runtimeData(rt(), 'Tempo esgotado em /runtime')} onClose={() => {}} />)
    expect(screen.getByRole('alert')).toHaveTextContent('exibindo a última consulta')
    expect(screen.getByTestId('tab-overview')).toBeInTheDocument()
  })
})

describe('integração na página', () => {
  beforeEach(() => {
    localStorage.clear()
    sessionStorage.clear()
    apiMock.cameras.mockResolvedValue([
      { id: 'c001', state: 'running', error: null, width: 1920, height: 1080, fps: 10, offset_s: 0, frames_total: 10, frame: 1, loop: 0, processing_fps: 8, inference_ms: 20, frames_published: 1, frames_dropped: 0, mode: 'live' },
    ])
    apiMock.health.mockResolvedValue({ status: 'ok', mode: 'live', environment: 't', device: 'cuda:0', uptime_s: 1, cameras: 1, ws_clients: 0, notes: [] })
    apiMock.runtime.mockReset()
    apiMock.runtime.mockResolvedValue(rt())
  })

  it('abre o painel sem tirar as câmeras da tela e só consulta /runtime com o painel aberto', async () => {
    const u = userEvent.setup()
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('panel-c001')).toBeInTheDocument())
    expect(apiMock.runtime).not.toHaveBeenCalled()
    await u.click(screen.getByTestId('open-processing'))
    await waitFor(() => expect(screen.getByTestId('processing-panel')).toBeInTheDocument())
    await waitFor(() => expect(screen.getByTestId('tab-overview')).toBeInTheDocument())
    expect(apiMock.runtime).toHaveBeenCalled()
    expect(screen.getByTestId('panel-c001')).toBeInTheDocument() // grade continua visível
    expect(screen.getByTestId('open-processing')).toHaveAttribute('aria-pressed', 'true')
    await u.click(screen.getByRole('button', { name: 'Fechar painel' }))
    // o painel continua montado (preserva aba e filtros), apenas oculto
    expect(screen.getByTestId('processing-panel')).toHaveAttribute('data-open', 'false')
    expect(screen.getByTestId('panel-c001')).toBeInTheDocument()
  })
})
