/** Contrato com o backend (mtmc/server.py). */
export type BackendCameraState =
  | 'starting' // processo iniciado, nenhum frame confirmado ainda
  | 'waiting' // vídeo ainda não começou na linha do tempo global
  | 'running' // frames confirmados recentemente
  | 'stalled' // já publicou, mas sem frames há mais de 3 s
  | 'error'
  | 'unavailable' // vídeo inacessível

export interface CameraStatus {
  id: string
  state: BackendCameraState
  error: string | null
  width: number
  height: number
  fps: number
  offset_s: number
  frames_total: number
  frame: number
  loop: number
  processing_fps: number
  inference_ms: number
  frames_published: number
  frames_dropped: number
  mode: 'live' | 'replay'
}

export interface Health {
  status: string
  mode: 'live' | 'replay'
  environment: string
  device: string
  uptime_s: number
  cameras: number
  ws_clients: number
  notes: string[]
  association?: { status?: string; linked_identities?: number; last_run_ms?: number; [k: string]: unknown }
}

/** Detecção local: `id` é o ID do ByteTrack (por câmera), NÃO uma identidade global MTMC. bbox em pixels do vídeo original. */
export interface Detection {
  id: number
  cls: number
  name: string
  conf: number
  bbox: [number, number, number, number]
  gid: string | null // ID global MTMC (só existe se houver associação salva)
  uncertain: boolean
}

export interface FrameHeader {
  cam: string
  seq: number
  frame: number
  loop: number
  t_global: number
  t_local: number
  width: number
  height: number
  stream_width: number
  stream_height: number
  proc_fps: number
  inference_ms: number
  mode: 'live' | 'replay'
  detections: Detection[]
}

export interface FramePacket {
  header: FrameHeader
  jpeg: Blob
}

export type StreamState = 'connecting' | 'waiting-frames' | 'streaming' | 'reconnecting' | 'closed'
export type LayoutId = '1' | '2' | '4' | '6' | '9' | 'auto'

/** GET /api/runtime — descrição SOMENTE LEITURA do que está em execução (mtmc/runtime_info.py). */
export type ParamStatus = 'confirmed' | 'unconfirmed' | 'diverges'

export interface RuntimeParam {
  id: string
  label: string
  category: string
  scope: 'global' | 'câmera' | 'modelo' | string
  configured: unknown // valor no arquivo de configuração (relido do disco)
  applied: unknown // valor lido do objeto em execução; null + status "unconfirmed" = não confirmado
  status: ParamStatus
  source: string // "runtime" (lido do objeto) | "config"
  description: string
  unit: string | null
  editable: boolean
  fixed_in_code: boolean
}

export interface RuntimeModelIO {
  inputs: { name: string; shape: (number | string)[]; dtype: string }[]
  outputs: { name: string; shape: (number | string)[]; dtype: string }[]
}

export interface RuntimeModel {
  id: string
  role: string
  name: string | null
  artifact: string | null
  task?: string | null
  format?: string | null
  backend_class?: string | null
  device?: string | null
  load_state?: string | null
  input_shape?: number[] | null
  io_dtype?: string | null
  io?: RuntimeModelIO
  engine_metadata?: { imgsz?: number[]; batch?: number; args?: Record<string, unknown>; version?: string; date?: string; description?: string }
  providers?: string[]
  runtime_version?: string
  stride?: number | null
  end2end?: boolean | null
  params?: number | null
  classes: { id: number; name: string }[]
  n_classes: number | null
  cameras: string[]
}

export interface RuntimeCameraParam {
  value: unknown
  origin: string
}

export interface RuntimeCamera {
  id: string
  state: string
  error: string | null
  model: string | null
  model_load: string | null
  tracker: string | null
  params: Record<string, RuntimeCameraParam>
}

export interface RuntimeInfo {
  schema_version: number
  read_only: boolean
  generated_at: string
  mode: 'live' | 'replay'
  started_at: string
  revision: string
  config: {
    file: string | null
    file_mtime: string | null
    sha256: string | null
    changed_on_disk: boolean | null
    diff: { key: string; running: unknown; on_disk: unknown }[]
    error?: string
  }
  versions: Record<string, string | null>
  history: { supported: boolean; reason?: string }
  models: RuntimeModel[]
  parameters: RuntimeParam[]
  cameras: RuntimeCamera[]
  association: Record<string, unknown>
}
