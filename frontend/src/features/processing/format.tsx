import { AlertTriangle, CheckCircle2, HelpCircle, Lock } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { cn } from '@/lib/utils'
import type { ParamStatus } from '@/types/api'

/** Valores com formatação consistente (números com até 3 casas, listas compactas). */
export function formatValue(v: unknown, unit?: string | null): string {
  if (v === null || v === undefined) return '—'
  if (typeof v === 'boolean') return v ? 'sim' : 'não'
  if (typeof v === 'number') {
    const s = Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')
    return unit ? `${s} ${unit}` : s
  }
  if (Array.isArray(v)) return `[${v.map((x) => formatValue(x)).join(', ')}]`
  if (typeof v === 'object') {
    const e = Object.entries(v as Record<string, unknown>)
    return e.length ? e.map(([k, x]) => `${k}: ${formatValue(x)}`).join(' · ') : '{}'
  }
  return String(v)
}

export const NOT_CONFIRMED = 'não confirmado pelo runtime'

export function ValueCell({ value, status, unit }: { value: unknown; status?: ParamStatus; unit?: string | null }) {
  if (status === 'unconfirmed') return <span className="italic text-muted-foreground">{NOT_CONFIRMED}</span>
  return <span className="break-all font-mono text-xs tabular-nums">{formatValue(value, unit)}</span>
}

const STATUS = {
  confirmed: { label: 'confirmado', cls: 'text-ok border-ok/40 bg-ok/10', Icon: CheckCircle2 },
  unconfirmed: { label: 'não confirmado', cls: 'text-muted-foreground border-border', Icon: HelpCircle },
  diverges: { label: 'diverge', cls: 'text-warn border-warn/50 bg-warn/10', Icon: AlertTriangle },
} as const

export function StatusBadge({ status }: { status: ParamStatus }) {
  const m = STATUS[status]
  return (
    <Badge variant="outline" className={cn('gap-1 px-1.5 text-[10px] font-medium', m.cls)} data-status={status}>
      <m.Icon className="size-3" aria-hidden />
      {m.label}
    </Badge>
  )
}

export function ScopeBadge({ scope }: { scope: string }) {
  return (
    <Badge variant="secondary" className="px-1.5 text-[10px] font-medium" data-scope={scope}>
      {scope}
    </Badge>
  )
}

export function FixedBadge() {
  return (
    <Badge variant="outline" className="gap-1 px-1.5 text-[10px] text-muted-foreground" title="Valor fixo no código ou derivado: não é parâmetro do YAML">
      <Lock className="size-3" aria-hidden />
      fixo no código
    </Badge>
  )
}

export function originLabel(origin: string): { text: string; inherited: boolean } {
  if (origin === 'global' || origin.startsWith('global')) return { text: 'herdado (global)', inherited: true }
  if (origin === 'camera') return { text: 'sobrescrito na câmera', inherited: false }
  return { text: origin, inherited: false }
}
