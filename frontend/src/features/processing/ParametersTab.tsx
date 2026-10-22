import { useMemo, useState } from 'react'
import { Search } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { FixedBadge, ScopeBadge, StatusBadge, ValueCell } from '@/features/processing/format'
import type { RuntimeInfo } from '@/types/api'

const ALL = 'todas'

function Sel({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: string[] }) {
  return (
    <label className="flex items-center gap-1 text-[11px] text-muted-foreground">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="h-7 rounded border bg-background px-1.5 text-xs text-foreground"
        aria-label={label}
      >
        <option value={ALL}>{ALL}</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}

export function ParametersTab({ rt }: { rt: RuntimeInfo }) {
  const [q, setQ] = useState('')
  const [cat, setCat] = useState(ALL)
  const [scope, setScope] = useState(ALL)
  const [status, setStatus] = useState(ALL)
  const cats = useMemo(() => [...new Set(rt.parameters.map((p) => p.category))], [rt.parameters])
  const scopes = useMemo(() => [...new Set(rt.parameters.map((p) => p.scope))], [rt.parameters])
  const rows = rt.parameters.filter(
    (p) =>
      (cat === ALL || p.category === cat) &&
      (scope === ALL || p.scope === scope) &&
      (status === ALL || p.status === status) &&
      (!q || `${p.label} ${p.id}`.toLowerCase().includes(q.toLowerCase())),
  )
  return (
    <div className="space-y-2" data-testid="tab-params">
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative">
          <Search className="pointer-events-none absolute left-2 top-2 size-3.5 text-muted-foreground" aria-hidden />
          <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Buscar parâmetro…" aria-label="Buscar parâmetro" className="h-7 w-44 pl-7 text-xs" />
        </div>
        <Sel label="categoria" value={cat} onChange={setCat} options={cats} />
        <Sel label="escopo" value={scope} onChange={setScope} options={scopes} />
        <Sel label="estado" value={status} onChange={setStatus} options={['confirmed', 'unconfirmed', 'diverges']} />
        <span className="ml-auto text-[11px] text-muted-foreground" data-testid="param-count">{rows.length} de {rt.parameters.length}</span>
      </div>
      <Table className="table-fixed">
        <TableHeader>
          <TableRow>
            <TableHead className="h-8 w-[34%] text-[11px]">Parâmetro</TableHead>
            <TableHead className="h-8 w-[20%] text-[11px]">Configurado (arquivo)</TableHead>
            <TableHead className="h-8 w-[20%] text-[11px]">Aplicado (runtime)</TableHead>
            <TableHead className="h-8 w-[12%] text-[11px]">Escopo</TableHead>
            <TableHead className="h-8 w-[14%] text-[11px]">Estado</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((p) => (
            <TableRow key={p.id} data-testid={`param-${p.id}`}>
              <TableCell className="whitespace-normal py-1.5 align-top">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <span className="cursor-help text-xs font-medium underline decoration-dotted underline-offset-2">{p.label}</span>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs text-xs">{p.description || 'sem descrição'}</TooltipContent>
                </Tooltip>
                <div className="font-mono text-[10px] text-muted-foreground">{p.id}</div>
              </TableCell>
              <TableCell className="whitespace-normal py-1.5 align-top"><ValueCell value={p.configured} unit={p.unit} /></TableCell>
              <TableCell className="whitespace-normal py-1.5 align-top"><ValueCell value={p.applied} status={p.status} unit={p.unit} /></TableCell>
              <TableCell className="whitespace-normal py-1.5 align-top">
                <div className="flex flex-col items-start gap-1">
                  <ScopeBadge scope={p.scope} />
                  {p.fixed_in_code && <FixedBadge />}
                </div>
              </TableCell>
              <TableCell className="whitespace-normal py-1.5 align-top"><StatusBadge status={p.status} /></TableCell>
            </TableRow>
          ))}
          {rows.length === 0 && (
            <TableRow>
              <TableCell colSpan={5} className="py-6 text-center text-xs text-muted-foreground">Nenhum parâmetro com esses filtros</TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </div>
  )
}
