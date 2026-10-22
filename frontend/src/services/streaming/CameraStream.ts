import { wsUrl } from '@/lib/config'
import type { FramePacket, StreamState } from '@/types/api'
import { parsePacket } from './protocol'

export interface StreamOptions {
  /** Sem mensagens por este tempo => fluxo considerado parado; reconecta. */
  stalledMs?: number
  backoffBaseMs?: number
  backoffMaxMs?: number
  /** Injetável para testes. */
  createSocket?: (url: string) => WebSocket
}

type PacketListener = (p: FramePacket) => void
type StateListener = (s: StreamState, info: { attempt: number }) => void

/** Uma conexão WebSocket por câmera, com reconexão exponencial e detecção de fluxo parado. */
export class CameraStream {
  private ws: WebSocket | null = null
  private state: StreamState = 'connecting'
  private attempt = 0
  private disposed = false
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private watchdog: ReturnType<typeof setInterval> | null = null
  private lastMsg = 0
  private packetL = new Set<PacketListener>()
  private stateL = new Set<StateListener>()
  private readonly stalledMs: number
  private readonly backoffBaseMs: number
  private readonly backoffMaxMs: number
  private readonly createSocket?: (url: string) => WebSocket

  readonly camId: string

  constructor(camId: string, opts: StreamOptions = {}) {
    this.camId = camId
    this.stalledMs = opts.stalledMs ?? 5000
    this.backoffBaseMs = opts.backoffBaseMs ?? 500
    this.backoffMaxMs = opts.backoffMaxMs ?? 8000
    this.createSocket = opts.createSocket
    this.connect()
  }

  get currentState() {
    return this.state
  }
  get attempts() {
    return this.attempt
  }

  onPacket(fn: PacketListener) {
    this.packetL.add(fn)
    return () => this.packetL.delete(fn)
  }
  onState(fn: StateListener) {
    this.stateL.add(fn)
    fn(this.state, { attempt: this.attempt })
    return () => this.stateL.delete(fn)
  }

  private setState(s: StreamState) {
    if (this.state === s) return
    this.state = s
    this.stateL.forEach((f) => f(s, { attempt: this.attempt }))
  }

  private connect() {
    if (this.disposed) return
    this.setState(this.attempt === 0 ? 'connecting' : 'reconnecting')
    const url = wsUrl(`/cameras/${encodeURIComponent(this.camId)}/ws`)
    const ws = this.createSocket ? this.createSocket(url) : new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws
    this.lastMsg = Date.now()
    ws.onopen = () => {
      if (this.ws !== ws) return
      this.lastMsg = Date.now()
      this.setState('waiting-frames')
    }
    ws.onmessage = (ev) => {
      if (this.ws !== ws || !(ev.data instanceof ArrayBuffer)) return
      try {
        const pkt = parsePacket(ev.data)
        this.lastMsg = Date.now()
        this.attempt = 0
        this.setState('streaming')
        this.packetL.forEach((f) => f(pkt))
      } catch {
        /* pacote inválido: ignora; o watchdog trata fluxo parado */
      }
    }
    ws.onclose = () => {
      if (this.ws === ws) this.scheduleRetry()
    }
    ws.onerror = () => ws.close()
    this.startWatchdog()
  }

  private startWatchdog() {
    if (this.watchdog) return
    this.watchdog = setInterval(
      () => {
        if (!this.disposed && !this.retryTimer && Date.now() - this.lastMsg > this.stalledMs) this.scheduleRetry()
      },
      Math.max(100, this.stalledMs / 4),
    )
  }

  private scheduleRetry() {
    if (this.disposed || this.retryTimer) return
    const old = this.ws
    this.ws = null
    if (old) {
      old.onopen = old.onmessage = old.onclose = old.onerror = null
      old.close()
    }
    this.attempt += 1
    this.setState('reconnecting')
    const delay = Math.min(this.backoffMaxMs, this.backoffBaseMs * 2 ** (this.attempt - 1))
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.connect()
    }, delay)
  }

  dispose() {
    this.disposed = true
    if (this.retryTimer) clearTimeout(this.retryTimer)
    if (this.watchdog) clearInterval(this.watchdog)
    const ws = this.ws
    this.ws = null
    if (ws) {
      ws.onopen = ws.onmessage = ws.onclose = ws.onerror = null
      ws.close()
    }
    this.setState('closed')
    this.packetL.clear()
    this.stateL.clear()
  }
}
