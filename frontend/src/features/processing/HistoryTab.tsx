import type { RuntimeInfo } from '@/types/api'

export function HistoryTab({ rt }: { rt: RuntimeInfo }) {
  return (
    <div className="space-y-2 text-sm" data-testid="tab-history">
      {rt.history.supported ? (
        <p>Histórico disponível.</p>
      ) : (
        <>
          <p className="font-medium">Sem histórico de alterações</p>
          <p className="text-xs text-muted-foreground">
            O backend não registra alterações de configuração: {rt.history.reason ?? 'recurso não implementado'}. Como o painel é somente leitura, não há
            usuário responsável nem valores anterior/novo para exibir. A única evidência de mudança disponível é a comparação entre o arquivo no disco e o
            que está em execução (aba Visão geral).
          </p>
          <p className="text-xs text-muted-foreground">
            Uma etapa de edição exigiria um contrato novo no backend (validação, persistência da alteração, confirmação do runtime e trilha de auditoria);
            a proposta está em <code className="font-mono">frontend/README.md</code>.
          </p>
        </>
      )}
    </div>
  )
}
