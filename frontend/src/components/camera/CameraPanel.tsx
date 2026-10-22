import { memo, useId, useRef } from 'react'
import { Boxes, Hash, Info, Maximize2, Minimize2, Route, Tag, X } from 'lucide-react'
import { DragHandle } from '@/components/camera/TileDragContext'
import { deriveHealth } from '@/components/status/CameraStateBadge'
import { Toggle } from '@/components/ui/toggle'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useCameraStream } from '@/hooks/useCameraStream'
import { useVisible } from '@/hooks/useVisible'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import { cn } from '@/lib/utils'
import type { CameraStatus } from '@/types/api'

interface Props {
  camera: CameraStatus
  overlay: OverlayOptions
  expanded: boolean
  /** Tile fora da tela por maximização/layout: a conexão é mantida, mas nada é decodificado nem desenhado. */
  hidden?: boolean
  onExpand: (id: string | null) => void
  onRemove: (id: string) => void
  onOverlayChange: (id: string, patch: Partial<OverlayOptions>) => void
  onInfo: (id: string) => void
}

const DOT: Record<string, string> = {
  live: 'bg-ok',
  connecting: 'bg-primary animate-pulse',
  reconnecting: 'bg-warn animate-pulse',
  waiting: 'bg-muted-foreground',
  stalled: 'bg-warn',
  error: 'bg-destructive',
  unavailable: 'bg-destructive',
  idle: 'bg-muted-foreground',
}
const HEALTH_TEXT: Record<string, string> = {
  live: 'Ao vivo',
  connecting: 'Conectando',
  reconnecting: 'Reconectando',
  waiting: 'Aguardando vídeo',
  stalled: 'Fluxo parado',
  error: 'Erro',
  unavailable: 'Indisponível',
  idle: '—',
}

function Message({ title, detail }: { title: string; detail?: string | null }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-1 bg-background/70 px-4 text-center" role="status">
      <p className="text-sm font-medium">{title}</p>
      {detail && <p className="max-w-full truncate text-xs text-muted-foreground">{detail}</p>}
    </div>
  )
}

function IconToggle({ label, pressed, onPressedChange, children, testId }: { label: string; pressed: boolean; onPressedChange: (v: boolean) => void; children: React.ReactNode; testId: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Toggle size="sm" pressed={pressed} onPressedChange={onPressedChange} aria-label={label} data-testid={testId} className="size-6 min-w-6 bg-transparent px-0 aria-pressed:bg-primary/25 aria-pressed:text-primary">
          {children}
        </Toggle>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  )
}

function IconButton({ label, onClick, children, testId }: { label: string; onClick: () => void; children: React.ReactNode; testId?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          aria-label={label}
          data-testid={testId}
          className="inline-flex size-6 items-center justify-center rounded text-foreground/90 hover:bg-white/15 focus-visible:bg-white/15"
        >
          {children}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  )
}

/**
 * Painel de uma câmera. Isolado e memoizado: só ele re-renderiza com o próprio fluxo (canvas imperativo).
 * O vídeo ocupa toda a área; identificação e controles ficam sobrepostos nas bordas (controles só no hover/foco).
 */
export const CameraPanel = memo(function CameraPanel({ camera, overlay, expanded, hidden = false, onExpand, onRemove, onOverlayChange, onInfo }: Props) {
  const box = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const instance = useId() // estável durante a vida do componente; muda só se for REMONTADO (usado nos testes)
  const visible = useVisible(box)
  const usable = camera.state !== 'unavailable'
  const connected = usable && (hidden || visible)
  const paint = !hidden && visible
  const info = useCameraStream(camera.id, connected, canvas, overlay, { paint })
  const health = deriveHealth(camera.state, connected ? info.state : null)
  const hasFrame = info.detections !== null

  let msg: { title: string; detail?: string | null } | null = null
  if (!usable) msg = { title: 'Vídeo indisponível', detail: camera.error }
  else if (camera.state === 'error') msg = { title: 'Erro na câmera', detail: camera.error }
  else if (info.state === 'reconnecting') msg = { title: 'Reconectando…', detail: `tentativa ${info.attempt}` }
  else if (!hasFrame && (info.state === 'connecting' || info.state === 'waiting-frames')) msg = { title: 'Conectando…' }
  else if (!hasFrame && camera.state === 'waiting') msg = { title: 'Aguardando início do vídeo' }
  else if (camera.state === 'stalled') msg = { title: 'Fluxo parado no backend', detail: 'sem frames novos há mais de 3 s' }

  const set = (patch: Partial<OverlayOptions>) => onOverlayChange(camera.id, patch)

  return (
    <section
      data-testid={`panel-${camera.id}`}
      data-instance={instance}
      data-health={health}
      aria-label={`Câmera ${camera.id}`}
      className="group relative flex h-full min-h-0 min-w-0 flex-col overflow-hidden rounded-md border border-border/80 bg-black"
    >
      <div ref={box} className="relative min-h-0 flex-1">
        <canvas ref={canvas} className="absolute inset-0 size-full object-contain" aria-label={`Vídeo ${camera.id}`} />
        {msg && <Message {...msg} />}
        {health === 'live' && info.detections === 0 && (
          <span className="absolute bottom-1.5 left-2 rounded bg-background/80 px-2 py-0.5 text-[11px] text-muted-foreground" data-testid="no-detections">
            Sem detecções neste frame
          </span>
        )}

        <div className="pointer-events-none absolute inset-x-0 top-0 flex items-start gap-1 p-1.5">
          <DragHandle label={camera.id}>
            <span className={cn('size-2 shrink-0 rounded-full', DOT[health])} aria-hidden />
            <span className="font-mono text-xs font-semibold leading-none">{camera.id}</span>
            <span className="sr-only" data-testid={`state-${camera.id}`} data-health={health}>{HEALTH_TEXT[health]}</span>
            {health !== 'live' && <span className="text-[10px] leading-none text-muted-foreground">{HEALTH_TEXT[health]}</span>}
          </DragHandle>

          <div
            data-no-dnd
            className={cn(
              'pointer-events-auto ml-auto flex items-center gap-0.5 rounded bg-black/65 p-0.5 backdrop-blur-sm',
              'opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 [@media(hover:none)]:opacity-100',
            )}
          >
            <IconToggle label="Caixas" pressed={overlay.boxes} onPressedChange={(v) => set({ boxes: v })} testId={`tgl-boxes-${camera.id}`}><Boxes className="size-3.5" /></IconToggle>
            <IconToggle label="IDs de tracking" pressed={overlay.ids} onPressedChange={(v) => set({ ids: v })} testId={`tgl-ids-${camera.id}`}><Hash className="size-3.5" /></IconToggle>
            <IconToggle label="Classe e confiança" pressed={overlay.labels} onPressedChange={(v) => set({ labels: v })} testId={`tgl-labels-${camera.id}`}><Tag className="size-3.5" /></IconToggle>
            <IconToggle label="Trilhas" pressed={overlay.trails} onPressedChange={(v) => set({ trails: v })} testId={`tgl-trails-${camera.id}`}><Route className="size-3.5" /></IconToggle>
            <span className="mx-0.5 h-4 w-px bg-white/20" aria-hidden />
            <IconButton label={`Configurações e parâmetros de ${camera.id}`} onClick={() => onInfo(camera.id)} testId={`info-${camera.id}`}><Info className="size-3.5" /></IconButton>
            <IconButton label={expanded ? `Voltar ao grid (${camera.id})` : `Maximizar ${camera.id}`} onClick={() => onExpand(expanded ? null : camera.id)} testId={`expand-${camera.id}`}>
              {expanded ? <Minimize2 className="size-3.5" /> : <Maximize2 className="size-3.5" />}
            </IconButton>
            <IconButton label={`Remover ${camera.id} do grid (o processamento continua)`} onClick={() => onRemove(camera.id)} testId={`remove-${camera.id}`}><X className="size-3.5" /></IconButton>
          </div>
        </div>
      </div>

      <footer className="flex h-[22px] shrink-0 items-center gap-3 border-t border-border/60 bg-card/70 px-2 font-mono text-[11px] tabular-nums text-muted-foreground" data-testid={`stats-${camera.id}`}>
        <span title="FPS recebido e exibido no navegador">{info.rxFps ? `${info.rxFps.toFixed(1)} fps` : '— fps'}</span>
        <span title="Objetos rastreados no frame">{info.detections === null ? '— obj' : `${info.detections} obj`}</span>
        <span title="Tempo de inferência no backend">{info.inferenceMs === null ? '— ms' : `${info.inferenceMs.toFixed(0)} ms`}</span>
        <span className="ml-auto" title="Tempo global (alinhado por cam_timestamp)">{info.tGlobal === null ? 't —' : `t ${info.tGlobal.toFixed(1)} s`}</span>
      </footer>
    </section>
  )
})
