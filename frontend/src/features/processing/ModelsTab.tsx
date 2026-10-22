import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { formatValue } from '@/features/processing/format'
import type { RuntimeInfo, RuntimeModel } from '@/types/api'

const NA = <span className="italic text-muted-foreground">não disponível</span>

function Field({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 border-b border-border/40 py-1 text-xs">
      <dt className="w-40 shrink-0 text-muted-foreground">{k}</dt>
      <dd className="min-w-0 break-all font-mono tabular-nums">{children ?? NA}</dd>
    </div>
  )
}

function precision(m: RuntimeModel) {
  const half = m.engine_metadata?.args?.half
  return (
    <>
      E/S: {m.io_dtype ?? NA}
      {half !== undefined && (
        <span className="text-muted-foreground"> · build: {half ? 'FP16' : 'FP32'} (metadado gravado no engine; não confirmado pelo runtime)</span>
      )}
      {m.io_dtype === undefined && m.engine_metadata === undefined && ' não disponível'}
    </>
  )
}

function ModelRow({ m }: { m: RuntimeModel }) {
  const [open, setOpen] = useState(false)
  const shape = m.input_shape ?? (m.io?.inputs?.[0]?.shape as unknown[] | undefined)
  return (
    <div className="rounded border" data-testid={`model-${m.id}`}>
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-accent/50"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        {open ? <ChevronDown className="size-4" aria-hidden /> : <ChevronRight className="size-4" aria-hidden />}
        <span className="font-mono text-sm font-semibold">{m.name ?? '—'}</span>
        <Badge variant="secondary" className="px-1.5 text-[10px]">{m.role}</Badge>
        <Badge variant="outline" className="px-1.5 text-[10px]">{m.format ?? 'formato não disponível'}</Badge>
        <Badge variant="outline" className="ml-auto px-1.5 text-[10px] text-ok border-ok/40">{m.load_state ?? 'estado não disponível'}</Badge>
      </button>
      {open && (
        <dl className="border-t px-3 py-2">
          <Field k="Tarefa">{m.task}</Field>
          <Field k="Artefato carregado">{m.artifact}</Field>
          <Field k="Formato (do runtime)">{m.format}</Field>
          <Field k="Backend (classe)">{m.backend_class ?? m.runtime_version}</Field>
          <Field k="Dimensões de entrada">{shape ? `[${shape.join(', ')}]` : null}</Field>
          <Field k="Precisão numérica">{precision(m)}</Field>
          <Field k="Dispositivo">{m.device}</Field>
          <Field k="Provedores (ONNX)">{m.providers?.join(', ')}</Field>
          <Field k="Parâmetros do modelo">{m.params != null ? m.params.toLocaleString('pt-BR') : null}</Field>
          <Field k="Metadados do engine">{m.engine_metadata ? formatValue({ imgsz: m.engine_metadata.imgsz, batch: m.engine_metadata.batch, versão: m.engine_metadata.version, data: m.engine_metadata.date }) : null}</Field>
          <Field k="Versão do modelo">
            <span className="italic text-muted-foreground">não disponível (os pesos não trazem versão; o formato não é inferido do nome do arquivo)</span>
          </Field>
          <Field k="Câmeras que o usam">{m.cameras.length ? m.cameras.join(', ') : null}</Field>
          {m.io && (
            <Field k="Tensores de E/S">
              {[...m.io.inputs.map((t) => `in  ${t.name} ${t.dtype} [${t.shape.join(',')}]`), ...m.io.outputs.map((t) => `out ${t.name} ${t.dtype} [${t.shape.join(',')}]`)].join('\n')}
            </Field>
          )}
          {m.role === 'detector' && (
            <div className="py-1 text-xs">
              <p className="mb-1 text-muted-foreground">Classes reconhecidas: {m.n_classes}</p>
              <div className="flex max-h-32 flex-wrap gap-1 overflow-auto" data-testid="class-list">
                {m.classes.map((c) => (
                  <Badge key={c.id} variant="outline" className="px-1.5 font-mono text-[10px]">
                    {c.id} · {c.name}
                  </Badge>
                ))}
              </div>
            </div>
          )}
        </dl>
      )}
    </div>
  )
}

export function ModelsTab({ rt }: { rt: RuntimeInfo }) {
  if (!rt.models.length) {
    return (
      <p className="text-sm text-muted-foreground" data-testid="tab-models">
        Nenhum modelo em execução{rt.mode === 'replay' ? ' (modo replay: o backend reproduz detecções salvas e não executa modelos)' : ''}.
      </p>
    )
  }
  return (
    <div className="space-y-2" data-testid="tab-models">
      {rt.models.map((m) => (
        <ModelRow key={m.id} m={m} />
      ))}
      <p className="pt-1 text-[11px] text-muted-foreground">
        Formato, dispositivo, entradas e provedores vêm de metadados do runtime (backend de inferência, engine TensorRT, sessão ONNX).
      </p>
    </div>
  )
}
