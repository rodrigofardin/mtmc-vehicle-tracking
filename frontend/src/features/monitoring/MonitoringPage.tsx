import { useEffect, useMemo, useState } from 'react'
import { VideoGrid } from '@/components/camera/VideoGrid'
import { CameraSidebar } from '@/components/layout/CameraSidebar'
import { TopBar } from '@/components/layout/TopBar'
import { Button } from '@/components/ui/button'
import { useSelection } from '@/features/camera-selection/useSelection'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import { useCameras } from '@/hooks/useCameras'
import { getConfig } from '@/lib/config'
import type { LayoutId } from '@/types/api'

export function MonitoringPage() {
  const { cameras, health, connection, error, refresh } = useCameras()
  const ids = useMemo(() => cameras.map((c) => c.id), [cameras])
  const { selected, toggle, selectAll, clear } = useSelection(ids)
  const [layout, setLayout] = useState<LayoutId>('auto')
  const [overlay, setOverlay] = useState<OverlayOptions>({ boxes: true, labels: true, trails: true })
  const [focusId, setFocusId] = useState<string | null>(null)

  const selectedCams = useMemo(() => cameras.filter((c) => selected.includes(c.id)), [cameras, selected])
  const processing = cameras.filter((c) => c.state === 'running').length

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setFocusId(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  let body
  if (connection === 'loading' && cameras.length === 0) {
    body = <Centered title="Carregando câmeras…" />
  } else if (connection === 'offline' && cameras.length === 0) {
    body = (
      <Centered title="Backend indisponível" detail={`${error ?? ''} — API: ${getConfig().API_BASE_URL}`}>
        <Button onClick={refresh}>Tentar novamente</Button>
      </Centered>
    )
  } else if (cameras.length === 0) {
    body = <Centered title="O backend não listou nenhuma câmera" />
  } else if (selectedCams.length === 0) {
    body = <Centered title="Nenhuma câmera selecionada" detail="Selecione câmeras na lista lateral para exibi-las." />
  } else {
    body = (
      <VideoGrid
        cameras={selectedCams}
        layout={layout}
        overlay={overlay}
        focusId={focusId}
        onFocus={setFocusId}
        onRemove={(id) => {
          toggle(id)
          setFocusId((f) => (f === id ? null : f))
        }}
      />
    )
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar
        environment={getConfig().ENVIRONMENT}
        connection={connection}
        health={health}
        total={cameras.length}
        selected={selectedCams.length}
        processing={processing}
        layout={layout}
        onLayout={(l) => {
          setLayout(l)
          setFocusId(null)
        }}
        overlay={overlay}
        onOverlay={setOverlay}
      />
      {connection === 'offline' && cameras.length > 0 && (
        <p className="bg-destructive/15 px-4 py-1 text-sm text-destructive" role="alert">
          Conexão com o backend perdida: {error}. Exibindo o último estado conhecido; reconectando automaticamente.
        </p>
      )}
      <div className="flex min-h-0 flex-1">
        <CameraSidebar cameras={cameras} selected={selected} onToggle={toggle} onSelectAll={selectAll} onClear={clear} />
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">{body}</main>
      </div>
    </div>
  )
}

function Centered({ title, detail, children }: { title: string; detail?: string; children?: React.ReactNode }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 p-8 text-center" role="status">
      <p className="text-lg font-medium">{title}</p>
      {detail && <p className="max-w-md text-sm text-muted-foreground">{detail}</p>}
      {children}
    </div>
  )
}
