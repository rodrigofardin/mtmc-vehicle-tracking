import type { FrameHeader } from '@/types/api'
import type { TrailStore } from './trails'

export interface OverlayOptions {
  boxes: boolean
  labels: boolean
  trails: boolean
}

export function colorFor(key: string | number): string {
  const s = String(key)
  let h = 0
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0
  return `hsl(${(h * 137.508) % 360}, 85%, 60%)`
}

export function labelFor(d: FrameHeader['detections'][number]): string {
  const parts = [`#${d.id}`, d.name, `${Math.round(d.conf * 100)}%`]
  if (d.gid) parts.push(d.gid)
  if (d.uncertain) parts.push('?')
  return parts.join(' ')
}

/** Desenha caixas, rótulos e trilhas. Coordenadas do backend (vídeo original) são escaladas para o canvas. */
export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  h: FrameHeader,
  trails: TrailStore,
  opts: OverlayOptions,
  canvasW: number,
) {
  const k = canvasW / h.width
  const lw = Math.max(1.5, canvasW / 480)
  const fs = Math.max(11, Math.round(canvasW / 62))
  ctx.lineJoin = 'round'
  ctx.font = `600 ${fs}px ui-sans-serif, system-ui, sans-serif`
  ctx.textBaseline = 'top'
  for (const d of h.detections) {
    const color = colorFor(d.gid ?? `${h.cam}:${d.id}`)
    if (opts.trails) {
      const pts = trails.get(d.id)
      if (pts.length > 1) {
        ctx.strokeStyle = color
        ctx.lineWidth = lw
        for (let i = 1; i < pts.length; i++) {
          ctx.globalAlpha = 0.25 + 0.75 * (i / pts.length) // mais antigo => mais transparente
          ctx.beginPath()
          ctx.moveTo(pts[i - 1].x * k, pts[i - 1].y * k)
          ctx.lineTo(pts[i].x * k, pts[i].y * k)
          ctx.stroke()
        }
        ctx.globalAlpha = 1
      }
    }
    const [x1, y1, x2, y2] = d.bbox.map((v) => v * k)
    if (opts.boxes) {
      ctx.strokeStyle = color
      ctx.lineWidth = lw
      ctx.setLineDash(d.uncertain ? [6, 4] : [])
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
      ctx.setLineDash([])
    }
    if (opts.labels) {
      const text = labelFor(d)
      const tw = ctx.measureText(text).width + 6
      const th = fs + 4
      const ty = y1 - th < 0 ? y1 : y1 - th
      ctx.fillStyle = color
      ctx.fillRect(x1, ty, tw, th)
      ctx.fillStyle = '#000'
      ctx.fillText(text, x1 + 3, ty + 2)
    }
  }
}
