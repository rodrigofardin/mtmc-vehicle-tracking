import { useMemo, useState } from 'react'
import { Badge } from '@/components/ui/badge'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { formatValue, originLabel } from '@/features/processing/format'
import { cn } from '@/lib/utils'
import type { RuntimeInfo } from '@/types/api'

const LABELS: Record<string, string> = {
  'detector.conf': 'Confiança mínima',
  'detector.imgsz': 'Entrada do detector',
  'tracker.track_buffer': 'Buffer do tracker (frames)',
  'tracker.match_thresh': 'Limiar de associação (IoU)',
  'video.fps': 'FPS do vídeo',
  'time.offset_s': 'Offset temporal (s)',
  'time.drift_ppm': 'Deriva (ppm)',
  'spatial.zones': 'Zonas (nº)',
  'spatial.homography': 'Homografia carregada',
}

export function CamerasTab({ rt }: { rt: RuntimeInfo }) {
  const [onlyDiff, setOnlyDiff] = useState(false)
  const keys = useMemo(() => Object.keys(rt.cameras[0]?.params ?? {}), [rt.cameras])
  const differs = (k: string) => new Set(rt.cameras.map((c) => JSON.stringify(c.params[k]?.value))).size > 1
  const shown = keys.filter((k) => !onlyDiff || differs(k))
  const summary: [string, (c: RuntimeInfo['cameras'][number]) => string][] = [
    ['Estado do processamento', (c) => c.state],
    ['Modelo ativo', (c) => c.model ?? 'não disponível'],
    ['Carregamento do modelo', (c) => c.model_load ?? 'não disponível'],
    ['Tracker', (c) => c.tracker ?? 'não disponível'],
  ]
  if (!rt.cameras.length) return <p className="text-sm text-muted-foreground">Nenhuma câmera.</p>
  const anyDiff = keys.some(differs)
  return (
    <div className="space-y-2" data-testid="tab-cameras">
      <div className="flex items-center gap-3 text-[11px] text-muted-foreground">
        <label className="flex cursor-pointer items-center gap-1.5">
          <input type="checkbox" checked={onlyDiff} onChange={(e) => setOnlyDiff(e.target.checked)} aria-label="Mostrar só divergências" />
          só divergências entre câmeras
        </label>
        <span data-testid="cams-diff-note">{anyDiff ? 'há parâmetros que diferem entre câmeras' : 'todas as câmeras usam os mesmos valores nos parâmetros de inferência/tracking'}</span>
      </div>
      <div className="overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="h-8 text-[11px]">Câmera →</TableHead>
              {rt.cameras.map((c) => (
                <TableHead key={c.id} className="h-8 font-mono text-[11px]">{c.id}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {summary.map(([label, get]) => (
              <TableRow key={label} className="bg-muted/30">
                <TableCell className="py-1.5 text-xs font-medium">{label}</TableCell>
                {rt.cameras.map((c) => (
                  <TableCell key={c.id} className="whitespace-normal py-1.5 font-mono text-xs">{get(c)}</TableCell>
                ))}
              </TableRow>
            ))}
            {shown.map((k) => (
              <TableRow key={k} data-testid={`camrow-${k}`} className={cn(differs(k) && 'bg-warn/5')}>
                <TableCell className="py-1.5 align-top text-xs">
                  {LABELS[k] ?? k}
                  <div className="font-mono text-[10px] text-muted-foreground">{k}</div>
                </TableCell>
                {rt.cameras.map((c) => {
                  const p = c.params[k]
                  const o = originLabel(p?.origin ?? '')
                  return (
                    <TableCell key={c.id} className="whitespace-normal py-1.5 align-top">
                      <div className="font-mono text-xs tabular-nums">{formatValue(p?.value)}</div>
                      <Badge variant={o.inherited ? 'outline' : 'secondary'} className="mt-0.5 px-1 text-[9px] font-normal">{o.text}</Badge>
                    </TableCell>
                  )
                })}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  )
}
