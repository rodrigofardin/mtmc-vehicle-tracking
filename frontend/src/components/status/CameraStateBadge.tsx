import { AlertTriangle, CircleDashed, Loader2, PauseCircle, Radio, WifiOff } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { BackendCameraState, StreamState } from '@/types/api'

export type Health = 'live' | 'connecting' | 'reconnecting' | 'waiting' | 'stalled' | 'error' | 'unavailable' | 'idle'

/**
 * Só marca "live" com confirmação dupla: o backend reporta `running` E o navegador está recebendo frames.
 */
export function deriveHealth(backend: BackendCameraState | undefined, stream: StreamState | null): Health {
  if (backend === 'unavailable') return 'unavailable'
  if (backend === 'error') return 'error'
  // sem fluxo no navegador (ex.: lista lateral): vale apenas o que o backend confirmou
  if (stream === null) {
    return backend === 'running' ? 'live' : backend === 'stalled' ? 'stalled' : backend === 'waiting' ? 'waiting' : 'connecting'
  }
  if (stream === 'reconnecting') return 'reconnecting'
  if (stream === 'connecting' || stream === 'waiting-frames') return 'connecting'
  if (backend === 'stalled') return 'stalled'
  if (backend === 'waiting' || backend === 'starting') return 'waiting'
  if (backend === 'running' && stream === 'streaming') return 'live'
  return 'connecting'
}

const META: Record<Health, { label: string; cls: string; Icon: typeof Radio; spin?: boolean }> = {
  live: { label: 'Ao vivo', cls: 'text-ok border-ok/40 bg-ok/10', Icon: Radio },
  connecting: { label: 'Conectando', cls: 'text-primary border-primary/40 bg-primary/10', Icon: Loader2, spin: true },
  reconnecting: { label: 'Reconectando', cls: 'text-warn border-warn/40 bg-warn/10', Icon: Loader2, spin: true },
  waiting: { label: 'Aguardando vídeo', cls: 'text-muted-foreground border-border', Icon: PauseCircle },
  stalled: { label: 'Fluxo parado', cls: 'text-warn border-warn/40 bg-warn/10', Icon: AlertTriangle },
  error: { label: 'Erro', cls: 'text-destructive border-destructive/40 bg-destructive/10', Icon: AlertTriangle },
  unavailable: { label: 'Indisponível', cls: 'text-destructive border-destructive/40 bg-destructive/10', Icon: WifiOff },
  idle: { label: 'Fora do grid', cls: 'text-muted-foreground border-border', Icon: CircleDashed },
}

export function CameraStateBadge({ health, className }: { health: Health; className?: string }) {
  const m = META[health]
  return (
    <Badge variant="outline" className={cn('gap-1 font-medium', m.cls, className)} data-health={health}>
      <m.Icon className={cn('size-3', m.spin && 'animate-spin')} aria-hidden />
      {m.label}
    </Badge>
  )
}
