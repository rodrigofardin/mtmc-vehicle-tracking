import { useState } from 'react'
import { RefreshCw, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { CamerasTab } from '@/features/processing/CamerasTab'
import { HistoryTab } from '@/features/processing/HistoryTab'
import { ModelsTab } from '@/features/processing/ModelsTab'
import { OverviewTab } from '@/features/processing/OverviewTab'
import { ParametersTab } from '@/features/processing/ParametersTab'
import type { PanelTab } from '@/features/monitoring/prefs'
import type { RuntimeData } from '@/hooks/useRuntime'

interface Props {
  runtime: RuntimeData
  onClose: () => void
  /** Painel sempre montado (preserva aba, filtros e rolagem); `open=false` apenas o oculta. */
  open?: boolean
  width?: number
  tab?: PanelTab
  onTab?: (t: PanelTab) => void
}

/** Painel lateral recolhível e redimensionável (não modal): as câmeras continuam visíveis ao lado. Somente leitura. */
export function ProcessingPanel({ runtime, onClose, open = true, width, tab = 'overview', onTab }: Props) {
  const { data, loading, error, refresh } = runtime
  // controlado pela página (aba persistida) ou, sem `onTab`, com estado próprio
  const [localTab, setLocalTab] = useState<PanelTab>(tab)
  const value = onTab ? tab : localTab
  const setTab = (t: PanelTab) => (onTab ? onTab(t) : setLocalTab(t))
  return (
    <aside
      className={cn(
        'flex shrink-0 flex-col border-l bg-card/95',
        'max-lg:absolute max-lg:inset-y-0 max-lg:right-0 max-lg:z-30 max-lg:shadow-2xl',
        !open && 'hidden',
      )}
      style={{ width: width ? `min(${width}px, 100vw)` : 'min(720px, 52vw)' }}
      aria-label="Modelos e limiares"
      aria-hidden={!open}
      data-testid="processing-panel"
      data-open={open}
    >
      <header className="flex items-center gap-2 border-b px-3 py-2">
        <h2 className="text-sm font-semibold">Modelos e limiares</h2>
        <Badge variant="outline" className="px-1.5 text-[10px]">somente leitura</Badge>
        {data && <Badge variant="secondary" className="px-1.5 font-mono text-[10px]" title="Revisão da configuração efetiva">rev {data.revision}</Badge>}
        <span className="ml-auto flex items-center gap-1">
          <Button size="icon" variant="ghost" className="size-7" onClick={refresh} aria-label="Atualizar" disabled={loading}>
            <RefreshCw className={loading ? 'size-4 animate-spin' : 'size-4'} />
          </Button>
          <Button size="icon" variant="ghost" className="size-7" onClick={onClose} aria-label="Fechar painel">
            <X className="size-4" />
          </Button>
        </span>
      </header>
      {error && (
        <p className="border-b bg-destructive/10 px-3 py-1.5 text-xs text-destructive" role="alert">
          {data ? 'Não foi possível atualizar; exibindo a última consulta: ' : 'Não foi possível consultar o backend: '}
          {error}
        </p>
      )}
      {!data && !error && <p className="p-4 text-sm text-muted-foreground" role="status">Carregando informações do processamento…</p>}
      {data && (
        <Tabs value={value} onValueChange={(v) => setTab(v as PanelTab)} className="flex min-h-0 flex-1 flex-col">
          <TabsList className="mx-3 mt-2 h-8 w-fit max-w-[calc(100%-1.5rem)] justify-start overflow-x-auto">
            <TabsTrigger value="overview" className="text-xs">Visão geral</TabsTrigger>
            <TabsTrigger value="models" className="text-xs">Modelos</TabsTrigger>
            <TabsTrigger value="params" className="text-xs">Limiares e parâmetros</TabsTrigger>
            <TabsTrigger value="cameras" className="text-xs">Por câmera</TabsTrigger>
            <TabsTrigger value="history" className="text-xs">Histórico</TabsTrigger>
          </TabsList>
          <div className="min-h-0 flex-1 overflow-y-auto overflow-x-hidden">
            <div className="p-3">
              <TabsContent value="overview"><OverviewTab rt={data} /></TabsContent>
              <TabsContent value="models"><ModelsTab rt={data} /></TabsContent>
              <TabsContent value="params"><ParametersTab rt={data} /></TabsContent>
              <TabsContent value="cameras"><CamerasTab rt={data} /></TabsContent>
              <TabsContent value="history"><HistoryTab rt={data} /></TabsContent>
            </div>
          </div>
        </Tabs>
      )}
    </aside>
  )
}
