import { CameraStream, type StreamOptions } from './CameraStream'

interface Entry {
  stream: CameraStream
  refs: number
  timer: ReturnType<typeof setTimeout> | null
}

/**
 * Compartilha UMA conexão por câmera (contagem de referências): sem conexões redundantes e sem
 * reinício ao mudar o layout. O fechamento é adiado (`graceMs`) para absorver remontagens.
 */
export class StreamManager {
  private streams = new Map<string, Entry>()
  private readonly opts: StreamOptions
  private readonly graceMs: number

  constructor(opts: StreamOptions = {}, graceMs = 1500) {
    this.opts = opts
    this.graceMs = graceMs
  }

  acquire(camId: string): { stream: CameraStream; release: () => void } {
    let e = this.streams.get(camId)
    if (!e) {
      e = { stream: new CameraStream(camId, this.opts), refs: 0, timer: null }
      this.streams.set(camId, e)
    }
    if (e.timer) {
      clearTimeout(e.timer)
      e.timer = null
    }
    e.refs += 1
    const entry = e
    let released = false
    return {
      stream: entry.stream,
      release: () => {
        if (released) return
        released = true
        entry.refs -= 1
        if (entry.refs <= 0 && !entry.timer) {
          entry.timer = setTimeout(() => {
            if (entry.refs <= 0) {
              entry.stream.dispose()
              this.streams.delete(camId)
            }
          }, this.graceMs)
        }
      },
    }
  }

  get openCount() {
    return this.streams.size
  }

  disposeAll() {
    this.streams.forEach((e) => e.stream.dispose())
    this.streams.clear()
  }
}

export const streamManager = new StreamManager()
