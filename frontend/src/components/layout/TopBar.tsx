import { Activity, Boxes, Route, Tag, Video, VideoOff } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { LAYOUTS } from '@/features/monitoring/layouts'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import type { BackendConnection } from '@/hooks/useCameras'
import { cn } from '@/lib/utils'
import type { Health, LayoutId } from '@/types/api'

interface Props {
  environment: string
  connection: BackendConnection
  health: Health | null
  total: number
  selected: number
  processing: number
  layout: LayoutId
  onLayout: (l: LayoutId) => void
  overlay: OverlayOptions
  onOverlay: (o: OverlayOptions) => void
}

const CONN = {
  loading: { label: 'Conectando ao backend…', cls: 'text-primary' },
  online: { label: 'Backend online', cls: 'text-ok' },
  offline: { label: 'Backend indisponível', cls: 'text-destructive' },
} as const

export function TopBar({ environment, connection, health, total, selected, processing, layout, onLayout, overlay, onOverlay }: Props) {
  const c = CONN[connection]
  return (
    <header className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b bg-card/60 px-4 py-2">
      <div className="flex items-center gap-2">
        <Activity className="size-5 text-primary" aria-hidden />
        <h1 className="text-base font-semibold tracking-tight">MTMC Monitor</h1>
        <Badge variant="outline" className="uppercase" data-testid="env">{health?.environment ?? environment}</Badge>
        {health && <Badge variant="secondary">{health.mode === 'live' ? `Ao vivo · ${health.device}` : 'Replay (resultados salvos)'}</Badge>}
      </div>

      <div className={cn('flex items-center gap-1.5 text-sm', c.cls)} role="status" data-testid="backend-status" data-connection={connection}>
        <span className="size-2 rounded-full bg-current" aria-hidden />
        {c.label}
      </div>

      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <span className="flex items-center gap-1" title="Câmeras disponíveis"><Video className="size-4" aria-hidden /><b data-testid="count-total" className="text-foreground">{total}</b> disponíveis</span>
        <span className="flex items-center gap-1" title="Câmeras selecionadas"><VideoOff className="size-4" aria-hidden /><b data-testid="count-selected" className="text-foreground">{selected}</b> selecionadas</span>
        <span title="Câmeras com processamento confirmado pelo backend"><b data-testid="count-processing" className="text-foreground">{processing}</b> processando</span>
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-1" role="group" aria-label="Sobreposições">
          <Toggle size="sm" variant="outline" pressed={overlay.boxes} onPressedChange={(v) => onOverlay({ ...overlay, boxes: v })} aria-label="Caixas"><Boxes className="size-4" /> Caixas</Toggle>
          <Toggle size="sm" variant="outline" pressed={overlay.labels} onPressedChange={(v) => onOverlay({ ...overlay, labels: v })} aria-label="Rótulos"><Tag className="size-4" /> Rótulos</Toggle>
          <Toggle size="sm" variant="outline" pressed={overlay.trails} onPressedChange={(v) => onOverlay({ ...overlay, trails: v })} aria-label="Trajetórias"><Route className="size-4" /> Trilhas</Toggle>
        </div>
        <ToggleGroup type="single" variant="outline" size="sm" value={layout} onValueChange={(v) => v && onLayout(v as LayoutId)} aria-label="Layout do grid">
          {LAYOUTS.map((l) => (
            <ToggleGroupItem key={l.id} value={l.id} aria-label={`Layout ${l.label}`} data-testid={`layout-${l.id}`}>
              {l.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </div>
    </header>
  )
}
