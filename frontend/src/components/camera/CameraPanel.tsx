import { memo, useRef } from 'react'
import { Maximize2, Minimize2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CameraStateBadge, deriveHealth } from '@/components/status/CameraStateBadge'
import { useCameraStream } from '@/hooks/useCameraStream'
import { useVisible } from '@/hooks/useVisible'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import type { CameraStatus } from '@/types/api'

interface Props {
  camera: CameraStatus
  overlay: OverlayOptions
  expanded: boolean
  onExpand: (id: string | null) => void
  onRemove: (id: string) => void
}

function Message({ title, detail }: { title: string; detail?: string | null }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-background/70 px-4 text-center" role="status">
      <p className="text-sm font-medium">{title}</p>
      {detail && <p className="max-w-full truncate text-xs text-muted-foreground">{detail}</p>}
    </div>
  )
}

/** Painel isolado de uma câmera: só ele re-renderiza com o próprio fluxo (memo + canvas imperativo). */
export const CameraPanel = memo(function CameraPanel({ camera, overlay, expanded, onExpand, onRemove }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const visible = useVisible(box)
  const usable = camera.state !== 'unavailable'
  const info = useCameraStream(camera.id, visible && usable, canvas, overlay)
  const health = deriveHealth(camera.state, visible && usable ? info.state : null)
  const hasFrame = info.detections !== null

  let msg: { title: string; detail?: string | null } | null = null
  if (!usable) msg = { title: 'Vídeo indisponível', detail: camera.error }
  else if (camera.state === 'error') msg = { title: 'Erro na câmera', detail: camera.error }
  else if (info.state === 'reconnecting') msg = { title: 'Reconectando…', detail: `tentativa ${info.attempt}` }
  else if (!hasFrame && (info.state === 'connecting' || info.state === 'waiting-frames')) msg = { title: 'Conectando…' }
  else if (!hasFrame && camera.state === 'waiting') msg = { title: 'Aguardando início do vídeo' }
  else if (camera.state === 'stalled') msg = { title: 'Fluxo parado no backend', detail: 'sem frames novos há mais de 3 s' }

  return (
    <section
      ref={box}
      data-testid={`panel-${camera.id}`}
      aria-label={`Câmera ${camera.id}`}
      className="group relative flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border bg-card"
    >
      <header className="flex items-center gap-2 border-b px-2 py-1.5 text-sm">
        <span className="font-mono font-semibold">{camera.id}</span>
        <CameraStateBadge health={health} />
        <span className="ml-auto flex items-center gap-1">
          <Button
            size="icon"
            variant="ghost"
            className="size-7"
            aria-label={expanded ? `Voltar ao grid (${camera.id})` : `Ampliar ${camera.id}`}
            onClick={() => onExpand(expanded ? null : camera.id)}
          >
            {expanded ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
          </Button>
          <Button size="icon" variant="ghost" className="size-7" aria-label={`Remover ${camera.id} do grid`} onClick={() => onRemove(camera.id)}>
            <X className="size-4" />
          </Button>
        </span>
      </header>
      <div className="relative min-h-0 flex-1 bg-black">
        <canvas ref={canvas} className="absolute inset-0 size-full object-contain" aria-label={`Vídeo ${camera.id}`} />
        {msg && <Message {...msg} />}
        {health === 'live' && info.detections === 0 && (
          <span className="absolute left-2 top-2 rounded bg-background/80 px-2 py-0.5 text-xs text-muted-foreground" data-testid="no-detections">
            Sem detecções neste frame
          </span>
        )}
      </div>
      <footer className="flex items-center gap-3 border-t px-2 py-1 font-mono text-[11px] text-muted-foreground" data-testid={`stats-${camera.id}`}>
        <span title="FPS recebido pelo navegador">{info.rxFps ? `${info.rxFps.toFixed(1)} fps` : '— fps'}</span>
        <span title="Objetos rastreados no frame">{info.detections === null ? '— obj' : `${info.detections} obj`}</span>
        <span title="Tempo de inferência no backend">{info.inferenceMs === null ? '— ms' : `${info.inferenceMs.toFixed(0)} ms`}</span>
        <span className="ml-auto" title="Tempo global (alinhado por cam_timestamp)">
          {info.tGlobal === null ? 't —' : `t ${info.tGlobal.toFixed(1)} s`}
        </span>
      </footer>
    </section>
  )
})
