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
