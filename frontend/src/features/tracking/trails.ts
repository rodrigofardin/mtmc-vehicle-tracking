import type { FrameHeader } from '@/types/api'

export interface Point {
  x: number
  y: number
}

interface Track {
  pts: Point[]
  lastFrame: number
}

/**
 * Histórico recente por ID de tracker, para UMA câmera (IDs locais de câmeras distintas nunca se misturam).
 * Memória limitada: no máximo `maxPoints` por trilha; trilhas sem observação por `maxMissingFrames` são descartadas.
 */
export class TrailStore {
  private tracks = new Map<number, Track>()
  private lastFrame = -1
  private lastLoop = -1

  private readonly maxPoints: number
  private readonly maxMissingFrames: number

  constructor(maxPoints = 40, maxMissingFrames = 30) {
    this.maxPoints = maxPoints
    this.maxMissingFrames = maxMissingFrames
  }

  update(h: Pick<FrameHeader, 'frame' | 'loop' | 'detections'>) {
    // vídeo reiniciou (novo loop) ou frame voltou no tempo => o ByteTrack foi reiniciado: IDs deixam de valer
    if (h.loop !== this.lastLoop || h.frame < this.lastFrame) this.tracks.clear()
    this.lastLoop = h.loop
    this.lastFrame = h.frame
    for (const d of h.detections) {
      const p = { x: (d.bbox[0] + d.bbox[2]) / 2, y: d.bbox[3] } // ponto de apoio (base da caixa)
      let t = this.tracks.get(d.id)
      if (!t) {
        t = { pts: [], lastFrame: h.frame }
        this.tracks.set(d.id, t)
      }
      t.pts.push(p)
      if (t.pts.length > this.maxPoints) t.pts.shift()
      t.lastFrame = h.frame
    }
    for (const [id, t] of this.tracks) if (h.frame - t.lastFrame > this.maxMissingFrames) this.tracks.delete(id)
  }

  get(id: number): readonly Point[] {
    return this.tracks.get(id)?.pts ?? []
  }

  get size() {
    return this.tracks.size
  }

  clear() {
    this.tracks.clear()
    this.lastFrame = -1
    this.lastLoop = -1
  }
}
