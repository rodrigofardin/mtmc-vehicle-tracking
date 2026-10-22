import { useMemo, useState } from 'react'
import { CheckCheck, PanelLeftClose, PanelLeftOpen, Search, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { CameraStateBadge, deriveHealth } from '@/components/status/CameraStateBadge'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'
import type { CameraStatus } from '@/types/api'

interface Props {
  cameras: CameraStatus[]
  selected: string[]
  collapsed: boolean
  width: number
  onToggleCollapsed: () => void
  onToggle: (id: string) => void
  onSelectAll: (ids: string[]) => void
  onClear: () => void // a confirmação (quando necessária) é responsabilidade de quem chama
}

const DOT: Record<string, string> = {
  live: 'bg-ok', connecting: 'bg-primary', waiting: 'bg-muted-foreground', stalled: 'bg-warn', reconnecting: 'bg-warn',
  error: 'bg-destructive', unavailable: 'bg-destructive', idle: 'bg-muted-foreground',
}

function RailButton({ label, onClick, children, testId }: { label: string; onClick: () => void; children: React.ReactNode; testId?: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button size="icon" variant="ghost" className="size-8" onClick={onClick} aria-label={label} data-testid={testId}>
          {children}
        </Button>
      </TooltipTrigger>
      <TooltipContent side="right">{label}</TooltipContent>
    </Tooltip>
  )
}

/** Lista dinâmica de câmeras. Desmarcar = remover da visualização (a inferência no backend continua). */
export function CameraSidebar({ cameras, selected, collapsed, width, onToggleCollapsed, onToggle, onSelectAll, onClear }: Props) {
  const [q, setQ] = useState('')
  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? cameras.filter((c) => c.id.toLowerCase().includes(s)) : cameras
  }, [cameras, q])

  if (collapsed) {
    return (
      <aside className="flex shrink-0 flex-col items-center gap-1 border-r bg-card/40 py-1.5" style={{ width: 56 }} aria-label="Câmeras (compacto)" data-testid="sidebar" data-collapsed="true">
        <RailButton label="Expandir lista de câmeras (B)" onClick={onToggleCollapsed} testId="sidebar-toggle"><PanelLeftOpen className="size-4" /></RailButton>
        <RailButton label="Buscar câmera (expande a lista)" onClick={onToggleCollapsed}><Search className="size-4" /></RailButton>
        <RailButton label="Selecionar todas" onClick={() => onSelectAll(cameras.map((c) => c.id))} testId="rail-select-all"><CheckCheck className="size-4" /></RailButton>
        <RailButton label="Limpar seleção" onClick={onClear} testId="rail-clear"><X className="size-4" /></RailButton>
        <div className="my-1 h-px w-8 bg-border" />
        <ul className="flex min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto" data-testid="camera-list-compact">
          {cameras.map((c) => {
            const isSel = selected.includes(c.id)
            const h = deriveHealth(c.state, null)
            return (
              <li key={c.id}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button
                      type="button"
                      onClick={() => onToggle(c.id)}
                      aria-pressed={isSel}
                      aria-label={`Câmera ${c.id}, ${isSel ? 'selecionada' : 'não selecionada'}`}
                      data-testid={`rail-cam-${c.id}`}
                      className={cn(
                        'relative flex h-8 w-10 items-center justify-center rounded border font-mono text-[11px] font-semibold',
                        isSel ? 'border-primary/70 bg-primary/15 text-primary' : 'border-border text-muted-foreground hover:bg-accent',
                      )}
                    >
                      {c.id.replace(/^c/i, '')}
                      <span className={cn('absolute right-0.5 top-0.5 size-1.5 rounded-full', DOT[h])} aria-hidden />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent side="right">{c.id} · {isSel ? 'no grid' : 'fora do grid'}</TooltipContent>
                </Tooltip>
              </li>
            )
          })}
        </ul>
      </aside>
    )
  }

  return (
    <aside className="flex shrink-0 flex-col border-r bg-card/40" style={{ width }} aria-label="Câmeras" data-testid="sidebar" data-collapsed="false">
      <div className="flex items-center gap-1 px-2 pb-1 pt-2">
        <h2 className="px-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Câmeras</h2>
        <span className="rounded bg-muted px-1.5 text-[10px] tabular-nums text-muted-foreground" data-testid="sidebar-count">{selected.length}/{cameras.length}</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button size="icon" variant="ghost" className="ml-auto size-7" onClick={onToggleCollapsed} aria-label="Recolher lista de câmeras (B)" data-testid="sidebar-toggle">
              <PanelLeftClose className="size-4" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="right">Recolher (B)</TooltipContent>
        </Tooltip>
      </div>
      <div className="space-y-1.5 px-2 pb-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-2 size-3.5 text-muted-foreground" aria-hidden />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Buscar câmera…" aria-label="Buscar câmera" className="h-7 pl-7 text-xs" />
        </div>
        <div className="flex gap-1.5">
          <Button size="sm" variant="secondary" className="h-7 flex-1 text-xs" onClick={() => onSelectAll(cameras.map((c) => c.id))}>
            Selecionar todas
          </Button>
          <Button size="sm" variant="outline" className="h-7 flex-1 text-xs" onClick={onClear}>
            Limpar
          </Button>
        </div>
        <p className="text-[10px] leading-tight text-muted-foreground">Desmarcar só remove do grid; o processamento no backend continua.</p>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">
        <ul className="space-y-0.5" data-testid="camera-list">
          {filtered.map((c) => {
            const isSel = selected.includes(c.id)
            return (
              <li key={c.id}>
                <label className="flex cursor-pointer items-center gap-2 rounded-md px-1.5 py-1.5 hover:bg-accent" data-testid={`cam-item-${c.id}`}>
                  <Checkbox checked={isSel} onCheckedChange={() => onToggle(c.id)} aria-label={`Selecionar ${c.id}`} />
                  <span className="min-w-0 flex-1">
                    <span className="block font-mono text-[13px] font-semibold leading-tight">{c.id}</span>
                    <span className="block truncate text-[10px] text-muted-foreground">
                      {c.state === 'unavailable' || c.state === 'error' ? (c.error ?? 'indisponível') : `${c.width}×${c.height} · ${c.processing_fps.toFixed(1)} fps proc.`}
                    </span>
                  </span>
                  <CameraStateBadge health={deriveHealth(c.state, null)} className="text-[10px]" />
                </label>
              </li>
            )
          })}
          {filtered.length === 0 && <li className="px-2 py-4 text-center text-xs text-muted-foreground">Nenhuma câmera encontrada</li>}
        </ul>
      </div>
    </aside>
  )
}
