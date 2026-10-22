import { useEffect, useRef, useState, type RefObject } from 'react'
import { drawOverlay, type OverlayOptions } from '@/features/tracking/drawOverlay'
import { TrailStore } from '@/features/tracking/trails'
import { streamManager, type StreamManager } from '@/services/streaming/StreamManager'
import type { FramePacket, StreamState } from '@/types/api'

export interface StreamInfo {
  state: StreamState
  attempt: number
  /** FPS efetivamente recebido pelo navegador (média móvel). */
  rxFps: number
  detections: number | null // null = nenhum frame recebido ainda
  tGlobal: number | null
  inferenceMs: number | null
  frame: number | null
}

const EMPTY: StreamInfo = { state: 'connecting', attempt: 0, rxFps: 0, detections: null, tGlobal: null, inferenceMs: null, frame: null }

/**
 * Liga um <canvas> ao fluxo de UMA câmera. Só o frame mais recente é decodificado/desenhado
 * (se o navegador não acompanha, os intermediários são descartados). `active=false` libera a conexão.
 */
export function useCameraStream(
  camId: string,
  active: boolean,
  canvasRef: RefObject<HTMLCanvasElement | null>,
  overlay: OverlayOptions,
  opts: { paint?: boolean } = {},
  manager: StreamManager = streamManager,
): StreamInfo {
  const [info, setInfo] = useState<StreamInfo>(EMPTY)
  const overlayRef = useRef(overlay)
  const paintOn = useRef(opts.paint !== false)
  useEffect(() => {
    overlayRef.current = overlay
    paintOn.current = opts.paint !== false // tile oculto (maximizado/outro layout): a conexão continua, só não decodifica nem desenha
  }, [overlay, opts.paint])

  useEffect(() => {
    if (!active) return
    const { stream, release } = manager.acquire(camId)
    const trails = new TrailStore()
    let pending: FramePacket | null = null
    let decoding = false
    let disposed = false
    let last = performance.now()
    let rxFps = 0
    let snapshot: StreamInfo = { ...EMPTY, state: stream.currentState }

    const flush = setInterval(() => setInfo(snapshot), 400) // atualização de UI limitada (2,5 Hz)

    const paint = async (pkt: FramePacket) => {
      decoding = true
      try {
        const bmp = await createImageBitmap(pkt.jpeg)
        const cv = canvasRef.current
        if (cv && !disposed) {
          if (cv.width !== bmp.width || cv.height !== bmp.height) {
            cv.width = bmp.width
            cv.height = bmp.height
          }
          const ctx = cv.getContext('2d')
          if (ctx) {
            ctx.drawImage(bmp, 0, 0)
            drawOverlay(ctx, pkt.header, trails, overlayRef.current, cv.width)
          }
        }
        bmp.close()
      } catch {
        /* frame corrompido: descarta */
      } finally {
        decoding = false
        if (pending && !disposed) {
          const p = pending
          pending = null
          void paint(p)
        }
      }
    }

    const offState = stream.onState((state, { attempt }) => {
      snapshot = { ...snapshot, state, attempt }
    })
    const offPacket = stream.onPacket((pkt) => {
      const now = performance.now()
      const inst = 1000 / Math.max(now - last, 1)
      last = now
      rxFps = rxFps ? rxFps * 0.85 + inst * 0.15 : inst
      const h = pkt.header
      trails.update(h) // dados sempre; desenho pode pular frames
      snapshot = {
        ...snapshot,
        rxFps,
        detections: h.detections.length,
        tGlobal: h.t_global,
        inferenceMs: h.inference_ms,
        frame: h.frame,
      }
      if (!paintOn.current) return
      if (decoding) pending = pkt
      else void paint(pkt)
    })

    return () => {
      disposed = true
      clearInterval(flush)
      offState()
      offPacket()
      release()
      trails.clear()
    }
  }, [camId, active, canvasRef, manager])

  return active ? info : EMPTY
}
