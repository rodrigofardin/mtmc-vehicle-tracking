import { AlertTriangle, FileText } from 'lucide-react'
import { formatValue } from '@/features/processing/format'
import type { RuntimeInfo } from '@/types/api'

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-3 border-b border-border/50 py-1.5 text-xs">
      <dt className="w-44 shrink-0 text-muted-foreground">{k}</dt>
      <dd className="min-w-0 break-all font-mono tabular-nums">{children}</dd>
    </div>
  )
}

export function OverviewTab({ rt }: { rt: RuntimeInfo }) {
  const counts = { confirmed: 0, unconfirmed: 0, diverges: 0 }
  rt.parameters.forEach((p) => (counts[p.status] += 1))
  const gpu = rt.versions.gpu
  return (
    <div className="space-y-4" data-testid="tab-overview">
      {rt.config.changed_on_disk && (
        <div className="rounded border border-warn/50 bg-warn/10 p-3 text-xs" role="alert" data-testid="config-changed">
          <p className="flex items-center gap-1.5 font-medium text-warn">
            <AlertTriangle className="size-4" aria-hidden /> O arquivo de configuração mudou depois da inicialização
          </p>
          <p className="mt-1 text-muted-foreground">
            {rt.config.diff.length} valor(es) no disco diferem do que está em execução. As mudanças só valem após reiniciar o backend
            (esta interface é somente leitura e não aplica nada).
          </p>
          <ul className="mt-2 space-y-0.5 font-mono">
            {rt.config.diff.slice(0, 8).map((d) => (
              <li key={d.key}>
                {d.key}: <span className="text-foreground">{formatValue(d.running)}</span> (rodando) → <span className="text-warn">{formatValue(d.on_disk)}</span> (disco)
              </li>
            ))}
          </ul>
        </div>
      )}

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Configuração</h3>
        <dl>
          <Row k="Modo">{rt.mode === 'live' ? 'ao vivo (inferência em execução)' : 'replay (resultados salvos; sem modelo)'}</Row>
          <Row k="Origem">
            <span className="inline-flex items-center gap-1">
              <FileText className="size-3" aria-hidden />
              {rt.config.file ?? 'não informada'}
            </span>
          </Row>
          <Row k="Revisão (hash do efetivo)">{rt.revision}</Row>
          <Row k="Arquivo: modificado em">{rt.config.file_mtime ?? '—'}</Row>
          <Row k="Arquivo: sha256 (16)">{rt.config.sha256 ?? '—'}</Row>
          <Row k="Backend iniciado em">{rt.started_at}</Row>
          <Row k="Consulta gerada em">{rt.generated_at}</Row>
          <Row k="Sincronização">
            {rt.config.changed_on_disk === null ? 'não verificável' : rt.config.changed_on_disk ? 'disco ≠ runtime (reinício necessário)' : 'disco = runtime'}
          </Row>
        </dl>
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Parâmetros</h3>
        <dl>
          <Row k="Total listados">{rt.parameters.length}</Row>
          <Row k="Confirmados pelo runtime">{counts.confirmed}</Row>
          <Row k="Não confirmados">{counts.unconfirmed}</Row>
          <Row k="Divergentes (config × runtime)">{counts.diverges}</Row>
        </dl>
      </section>

      <section>
        <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">Runtime de inferência</h3>
        <dl>
          {Object.entries(rt.versions)
            .filter(([k]) => k !== 'gpu')
            .map(([k, v]) => (
              <Row key={k} k={k}>
                {v ?? 'indisponível'}
              </Row>
            ))}
          <Row k="GPU">{gpu ?? 'indisponível'}</Row>
        </dl>
      </section>

      <p className="text-[11px] text-muted-foreground">
        Painel somente leitura. “Configurado” é o valor do arquivo relido do disco; “aplicado” é lido do objeto em execução. Onde o runtime não expõe o
        valor, aparece “não confirmado pelo runtime”.
      </p>
    </div>
  )
}
