import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import { CameraStateBadge, deriveHealth } from '@/components/status/CameraStateBadge'
import type { CameraStatus } from '@/types/api'

interface Props {
  cameras: CameraStatus[]
  selected: string[]
  onToggle: (id: string) => void
  onSelectAll: (ids: string[]) => void
  onClear: () => void
}

/** Lista dinâmica de câmeras. Desmarcar = remover da visualização (a inferência no backend continua). */
export function CameraSidebar({ cameras, selected, onToggle, onSelectAll, onClear }: Props) {
  const [q, setQ] = useState('')
  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    return s ? cameras.filter((c) => c.id.toLowerCase().includes(s)) : cameras
  }, [cameras, q])

  return (
    <aside className="flex w-64 shrink-0 flex-col border-r bg-card/40" aria-label="Câmeras">
      <div className="space-y-2 p-3">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-2.5 size-4 text-muted-foreground" aria-hidden />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Buscar câmera…" aria-label="Buscar câmera" className="pl-8" />
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="secondary" className="flex-1" onClick={() => onSelectAll(cameras.map((c) => c.id))}>
            Selecionar todas
          </Button>
          <Button size="sm" variant="outline" className="flex-1" onClick={onClear}>
            Limpar
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Desmarcar só remove do grid; o processamento no backend continua.
        </p>
      </div>
      <ScrollArea className="min-h-0 flex-1">
        <ul className="space-y-1 px-2 pb-3" data-testid="camera-list">
          {filtered.map((c) => {
            const isSel = selected.includes(c.id)
            return (
              <li key={c.id}>
                <label className="flex cursor-pointer items-center gap-2 rounded-md px-2 py-2 hover:bg-accent" data-testid={`cam-item-${c.id}`}>
                  <Checkbox checked={isSel} onCheckedChange={() => onToggle(c.id)} aria-label={`Selecionar ${c.id}`} />
                  <span className="min-w-0 flex-1">
                    <span className="block font-mono text-sm font-semibold">{c.id}</span>
                    <span className="block truncate text-[11px] text-muted-foreground">
                      {c.state === 'unavailable' || c.state === 'error'
                        ? (c.error ?? 'indisponível')
                        : `${c.width}×${c.height} · ${c.processing_fps.toFixed(1)} fps proc.`}
                    </span>
                  </span>
                  <CameraStateBadge health={deriveHealth(c.state, null)} className="text-[10px]" />
                </label>
              </li>
            )
          })}
          {filtered.length === 0 && <li className="px-2 py-4 text-center text-sm text-muted-foreground">Nenhuma câmera encontrada</li>}
        </ul>
      </ScrollArea>
    </aside>
  )
}
