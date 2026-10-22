import { Activity, AlertTriangle, Boxes, Gauge, Hash, Keyboard, Route, SlidersHorizontal, Tag, Video } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { LAYOUTS } from '@/features/monitoring/layouts'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import type { BackendConnection } from '@/hooks/useCameras'
import { cn } from '@/lib/utils'
import type { Health, LayoutId } from '@/types/api'

export interface TopStats {
  total: number
  selected: number
  connected: number // câmeras com processamento confirmado pelo backend
  failed: number // erro / indisponível / fluxo parado
  avgFps: number | null // média do FPS de PROCESSAMENTO reportado pelo backend (câmeras ativas)
  avgMs: number | null // média da latência de inferência reportada pelo backend
}

interface Props {
  environment: string
  connection: BackendConnection
  health: Health | null
  stats: TopStats
  layout: LayoutId
  onLayout: (l: LayoutId) => void
  overlay: OverlayOptions
  onOverlay: (patch: Partial<OverlayOptions>) => void
  panelOpen: boolean
  onTogglePanel: () => void
  helpOpen: boolean
  onHelpOpen: (o: boolean) => void
}

const CONN = {
  loading: { label: 'Conectando…', dot: 'bg-primary animate-pulse', text: 'text-primary' },
  online: { label: 'Backend online', dot: 'bg-ok', text: 'text-ok' },
  offline: { label: 'Backend indisponível', dot: 'bg-destructive', text: 'text-destructive' },
} as const

export const SHORTCUTS: [string, string][] = [
  ['1  2  4  6  9', 'Layout com 1, 2, 4, 6 ou 9 câmeras'],
  ['A', 'Layout automático'],
  ['B', 'Recolher/expandir a lista de câmeras'],
  ['P', 'Abrir/fechar o painel de modelos e limiares'],
  ['Esc', 'Sair da câmera maximizada'],
  ['?', 'Mostrar esta ajuda'],
  ['Espaço + setas', 'Reordenar a câmera focada pelo teclado'],
]

function Metric({ icon, children, tip, className, testId }: { icon: React.ReactNode; children: React.ReactNode; tip: string; className?: string; testId?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className={cn('inline-flex cursor-default items-center gap-1 text-xs text-muted-foreground', className)} data-testid={testId} tabIndex={0}>
          {icon}
          {children}
        </span>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-xs whitespace-pre-line">{tip}</TooltipContent>
    </Tooltip>
  )
}

function OverlayToggle({ label, pressed, onChange, children, testId }: { label: string; pressed: boolean; onChange: (v: boolean) => void; children: React.ReactNode; testId: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Toggle size="sm" variant="outline" pressed={pressed} onPressedChange={onChange} aria-label={label} data-testid={testId} className="size-7 min-w-7 px-0 aria-pressed:border-primary/50 aria-pressed:bg-primary/15 aria-pressed:text-primary">
          {children}
        </Toggle>
      </TooltipTrigger>
      <TooltipContent side="bottom">{label}</TooltipContent>
    </Tooltip>
  )
}

export function TopBar({ environment, connection, health, stats, layout, onLayout, overlay, onOverlay, panelOpen, onTogglePanel, helpOpen, onHelpOpen }: Props) {
  const c = CONN[connection]
  const assoc = health?.association as { status?: string; linked_identities?: number; last_run_ms?: number } | undefined
  const perfTip = [
    stats.avgFps !== null ? `Processamento médio: ${stats.avgFps.toFixed(1)} fps por câmera ativa (reportado pelo backend)` : 'FPS de processamento: indisponível',
    stats.avgMs !== null ? `Inferência média: ${stats.avgMs.toFixed(0)} ms por frame (inclui espera pela GPU compartilhada)` : null,
    health ? `Modo: ${health.mode === 'live' ? 'ao vivo' : 'replay'} · dispositivo: ${health.device}` : null,
    health ? `Clientes WebSocket: ${health.ws_clients} · uptime: ${Math.round(health.uptime_s / 60)} min` : null,
    assoc?.status ? `Associação global: ${assoc.status}${assoc.linked_identities != null ? ` · ${assoc.linked_identities} identidades ligadas` : ''}${assoc.last_run_ms ? ` · ${(assoc.last_run_ms / 1000).toFixed(1)} s/rodada` : ''}` : null,
  ].filter(Boolean).join('\n')

  return (
    <header className="flex min-h-11 flex-wrap items-center gap-x-3 gap-y-1 border-b bg-card/70 px-3 py-1">
      <div className="flex items-center gap-2">
        <Activity className="size-4 text-primary" aria-hidden />
        <h1 className="text-sm font-semibold tracking-tight">MTMC Monitor</h1>
        <Badge variant="outline" className="h-5 px-1.5 text-[10px] uppercase" data-testid="env">{health?.environment ?? environment}</Badge>
        {health && <Badge variant="secondary" className="h-5 px-1.5 text-[10px]">{health.mode === 'live' ? `ao vivo · ${health.device}` : 'replay'}</Badge>}
      </div>

      <span className="hidden h-4 w-px bg-border sm:block" aria-hidden />

      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5">
        <span className={cn('inline-flex items-center gap-1.5 text-xs font-medium', c.text)} role="status" data-testid="backend-status" data-connection={connection}>
          <span className={cn('size-2 rounded-full', c.dot)} aria-hidden />
          {c.label}
        </span>
        <Metric icon={<Video className="size-3.5" aria-hidden />} tip="Câmeras conectadas (processamento confirmado) / disponíveis no backend" testId="metric-connected">
          <b className="text-foreground tabular-nums" data-testid="count-processing">{stats.connected}</b>/<b className="text-foreground tabular-nums" data-testid="count-total">{stats.total}</b>
        </Metric>
        <Metric
          icon={<AlertTriangle className={cn('size-3.5', stats.failed > 0 && 'text-destructive')} aria-hidden />}
          tip="Câmeras com falha: erro, vídeo indisponível ou fluxo parado"
          className={stats.failed > 0 ? 'text-destructive' : undefined}
          testId="metric-failed"
        >
          <b className={cn('tabular-nums', stats.failed > 0 ? 'text-destructive' : 'text-foreground')} data-testid="count-failed">{stats.failed}</b> falha{stats.failed === 1 ? '' : 's'}
        </Metric>
        <Metric icon={<Hash className="size-3.5" aria-hidden />} tip="Câmeras selecionadas para exibição no grid (desmarcar não interrompe o processamento)">
          <b className="text-foreground tabular-nums" data-testid="count-selected">{stats.selected}</b> no grid
        </Metric>
        <Metric icon={<Gauge className="size-3.5" aria-hidden />} tip={perfTip} testId="metric-perf">
          {stats.avgFps !== null ? <><b className="text-foreground tabular-nums">{stats.avgFps.toFixed(1)}</b> fps</> : '— fps'}
          <span aria-hidden>·</span>
          {stats.avgMs !== null ? <><b className="text-foreground tabular-nums">{stats.avgMs.toFixed(0)}</b> ms</> : '— ms'}
        </Metric>
      </div>

      <div className="ml-auto flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-1" role="group" aria-label="Sobreposições (todas as câmeras)">
          <OverlayToggle label="Caixas (todas as câmeras)" pressed={overlay.boxes} onChange={(v) => onOverlay({ boxes: v })} testId="global-boxes"><Boxes className="size-3.5" /></OverlayToggle>
          <OverlayToggle label="IDs de tracking (todas)" pressed={overlay.ids} onChange={(v) => onOverlay({ ids: v })} testId="global-ids"><Hash className="size-3.5" /></OverlayToggle>
          <OverlayToggle label="Classe e confiança (todas)" pressed={overlay.labels} onChange={(v) => onOverlay({ labels: v })} testId="global-labels"><Tag className="size-3.5" /></OverlayToggle>
          <OverlayToggle label="Trilhas (todas as câmeras)" pressed={overlay.trails} onChange={(v) => onOverlay({ trails: v })} testId="global-trails"><Route className="size-3.5" /></OverlayToggle>
        </div>
        <ToggleGroup type="single" variant="outline" size="sm" value={layout} onValueChange={(v) => v && onLayout(v as LayoutId)} aria-label="Layout do grid">
          {LAYOUTS.map((l) => (
            <ToggleGroupItem key={l.id} value={l.id} aria-label={`Layout ${l.label}`} data-testid={`layout-${l.id}`} className="h-7 min-w-7 px-1.5 text-xs">
              {l.label}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <Button size="sm" variant={panelOpen ? 'secondary' : 'outline'} className="h-7 text-xs" onClick={onTogglePanel} aria-pressed={panelOpen} data-testid="open-processing">
          <SlidersHorizontal className="size-3.5" aria-hidden /> <span className="max-sm:hidden">Modelos e limiares</span>
        </Button>
        <Popover open={helpOpen} onOpenChange={onHelpOpen}>
          <Tooltip>
            <TooltipTrigger asChild>
              <PopoverTrigger asChild>
                <Button size="icon" variant="ghost" className="size-7" aria-label="Atalhos de teclado (?)" data-testid="open-help"><Keyboard className="size-4" /></Button>
              </PopoverTrigger>
            </TooltipTrigger>
            <TooltipContent side="bottom">Atalhos (?)</TooltipContent>
          </Tooltip>
          <PopoverContent align="end" className="w-80 text-xs" data-testid="help-popover">
            <p className="mb-2 font-semibold">Atalhos de teclado</p>
            <ul className="space-y-1">
              {SHORTCUTS.map(([k, d]) => (
                <li key={k} className="flex gap-3">
                  <kbd className="w-28 shrink-0 rounded border bg-muted px-1.5 py-0.5 text-center font-mono text-[10px]">{k}</kbd>
                  <span className="text-muted-foreground">{d}</span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-[10px] text-muted-foreground">Os atalhos ficam suspensos ao digitar em campos e com diálogos abertos.</p>
          </PopoverContent>
        </Popover>
      </div>
    </header>
  )
}
