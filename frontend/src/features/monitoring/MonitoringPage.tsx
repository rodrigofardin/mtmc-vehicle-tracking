import { useCallback, useEffect, useMemo, useState } from 'react'
import { PanelRightOpen } from 'lucide-react'
import { VideoGrid } from '@/components/camera/VideoGrid'
import { CameraSidebar } from '@/components/layout/CameraSidebar'
import { Splitter } from '@/components/layout/Splitter'
import { TopBar, type TopStats } from '@/components/layout/TopBar'
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Button } from '@/components/ui/button'
import { mergeOrder, swapInOrder, visibleInOrder } from '@/features/camera-selection/order'
import { useSelection } from '@/features/camera-selection/useSelection'
import { LIMITS, PREFS_KEY, type PanelTab, type UiPrefs } from '@/features/monitoring/prefs'
import { ProcessingPanel } from '@/features/processing/ProcessingPanel'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import { useCameras } from '@/hooks/useCameras'
import { useRuntime } from '@/hooks/useRuntime'
import { modalOpen, isTypingTarget, useShortcuts } from '@/hooks/useShortcuts'
import { useUiPrefs } from '@/hooks/useUiPrefs'
import { getConfig } from '@/lib/config'
import type { LayoutId } from '@/types/api'

const KEYS = ['boxes', 'ids', 'labels', 'trails'] as const

export function MonitoringPage() {
  const { cameras, health, connection, error, refresh } = useCameras()
  const idsKey = cameras.map((c) => c.id).join('|')
  const ids = useMemo(() => (idsKey ? idsKey.split('|') : []), [idsKey])
  const { selected, toggle, selectAll, clear } = useSelection(ids)
  const [prefs, updatePrefs] = useUiPrefs()
  const [focusId, setFocusId] = useState<string | null>(null)
  const [confirmClear, setConfirmClear] = useState(false)
  const [helpOpen, setHelpOpen] = useState(false)
  const runtime = useRuntime(prefs.panel.open)
  const patch = useCallback((fn: (p: UiPrefs) => UiPrefs) => updatePrefs(fn), [updatePrefs])

  // Telas estreitas, sem preferência salva: começa com a lista de câmeras recolhida
  useEffect(() => {
    try {
      if (window.innerWidth < 1024 && !localStorage.getItem(PREFS_KEY)) patch((p) => ({ ...p, sidebar: { ...p.sidebar, collapsed: true } }))
    } catch {
      /* sem armazenamento: mantém o padrão */
    }
  }, [patch])

  // ----- ordem: persistida (todas as câmeras conhecidas); o grid mostra ordem ∩ selecionadas
  const order = useMemo(() => mergeOrder(prefs.order, ids), [prefs.order, ids])
  const byId = useMemo(() => new Map(cameras.map((c) => [c.id, c])), [cameras])
  const selectedCams = useMemo(() => visibleInOrder(order, selected).map((id) => byId.get(id)!).filter(Boolean), [order, selected, byId])
  const onSwap = useCallback(
    (a: string, b: string) =>
      patch((p) => {
        const next = swapInOrder(mergeOrder(p.order, ids), a, b)
        return { ...p, order: [...next, ...p.order.filter((x) => !next.includes(x))] } // ids que sumiram ficam salvos
      }),
    [patch, ids],
  )

  // ----- sobreposições: padrão global + sobrescritas por câmera (valor efetivo estável por câmera)
  const overlayMap = useMemo(() => {
    const m: Record<string, OverlayOptions> = {}
    for (const id of ids) m[id] = { ...prefs.overlay, ...prefs.overlayOverrides[id] }
    return m
  }, [ids, prefs.overlay, prefs.overlayOverrides])
  const overlayFor = useCallback((id: string) => overlayMap[id] ?? prefs.overlay, [overlayMap, prefs.overlay])
  const onOverlayChange = useCallback(
    (id: string, change: Partial<OverlayOptions>) =>
      patch((p) => {
        const cur: Partial<OverlayOptions> = { ...p.overlayOverrides[id], ...change }
        for (const k of KEYS) if (cur[k] === p.overlay[k]) delete cur[k] // igual ao padrão global: deixa de ser sobrescrita
        const ov = { ...p.overlayOverrides }
        if (Object.keys(cur).length) ov[id] = cur
        else delete ov[id]
        return { ...p, overlayOverrides: ov }
      }),
    [patch],
  )
  const onGlobalOverlay = useCallback(
    (change: Partial<OverlayOptions>) =>
      patch((p) => {
        const ov: UiPrefs['overlayOverrides'] = {}
        for (const [id, o] of Object.entries(p.overlayOverrides)) {
          const r = { ...o }
          for (const k of Object.keys(change) as (keyof OverlayOptions)[]) delete r[k] // "todas as câmeras" substitui sobrescritas dessa chave
          if (Object.keys(r).length) ov[id] = r
        }
        return { ...p, overlay: { ...p.overlay, ...change }, overlayOverrides: ov }
      }),
    [patch],
  )

  // ----- navegação
  const setLayout = useCallback((l: LayoutId) => { patch((p) => ({ ...p, layout: l })); setFocusId(null) }, [patch])
  const toggleSidebar = useCallback(() => patch((p) => ({ ...p, sidebar: { ...p.sidebar, collapsed: !p.sidebar.collapsed } })), [patch])
  const setPanel = useCallback((open: boolean, tab?: PanelTab) => patch((p) => ({ ...p, panel: { ...p.panel, open, tab: tab ?? p.panel.tab } })), [patch])
  const togglePanel = useCallback(() => patch((p) => ({ ...p, panel: { ...p.panel, open: !p.panel.open } })), [patch])
  const onInfo = useCallback(() => setPanel(true, 'cameras'), [setPanel])
  const onRemove = useCallback((id: string) => { toggle(id); setFocusId((f) => (f === id ? null : f)) }, [toggle])
  const requestClear = useCallback(() => { if (selected.length > 1) setConfirmClear(true); else clear() }, [selected.length, clear])

  useShortcuts({
    '1': () => setLayout('1'), '2': () => setLayout('2'), '4': () => setLayout('4'), '6': () => setLayout('6'), '9': () => setLayout('9'),
    a: () => setLayout('auto'), b: toggleSidebar, p: togglePanel, '?': () => setHelpOpen((o) => !o),
  })
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape' || e.defaultPrevented || isTypingTarget(e.target) || modalOpen()) return
      setFocusId(null)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const stats: TopStats = useMemo(() => {
    const running = cameras.filter((c) => c.state === 'running')
    const fps = running.map((c) => c.processing_fps).filter((v) => v > 0)
    const ms = running.map((c) => c.inference_ms).filter((v) => v > 0)
    const avg = (a: number[]) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null)
    return {
      total: cameras.length, selected: selectedCams.length, connected: running.length,
      failed: cameras.filter((c) => ['error', 'unavailable', 'stalled'].includes(c.state)).length,
      avgFps: avg(fps), avgMs: avg(ms),
    }
  }, [cameras, selectedCams.length])

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
        layout={prefs.layout}
        overlayFor={overlayFor}
        focusId={focusId}
        onFocus={setFocusId}
        onRemove={onRemove}
        onOverlayChange={onOverlayChange}
        onInfo={onInfo}
        onSwap={onSwap}
      />
    )
  }

  const { sidebar, panel } = prefs
  return (
    <div className="flex h-full flex-col">
      <TopBar
        environment={getConfig().ENVIRONMENT}
        connection={connection}
        health={health}
        stats={stats}
        layout={prefs.layout}
        onLayout={setLayout}
        overlay={prefs.overlay}
        onOverlay={onGlobalOverlay}
        panelOpen={panel.open}
        onTogglePanel={togglePanel}
        helpOpen={helpOpen}
        onHelpOpen={setHelpOpen}
      />
      {connection === 'offline' && cameras.length > 0 && (
        <p className="bg-destructive/15 px-4 py-1 text-sm text-destructive" role="alert">
          Conexão com o backend perdida: {error}. Exibindo o último estado conhecido; reconectando automaticamente.
        </p>
      )}
      <div className="relative flex min-h-0 flex-1">
        <CameraSidebar
          cameras={cameras}
          selected={selected}
          collapsed={sidebar.collapsed}
          width={sidebar.width}
          onToggleCollapsed={toggleSidebar}
          onToggle={toggle}
          onSelectAll={selectAll}
          onClear={requestClear}
        />
        {!sidebar.collapsed && (
          <Splitter
            mode="grow-right"
            label="Redimensionar lista de câmeras"
            width={sidebar.width}
            min={LIMITS.sidebar.min}
            max={LIMITS.sidebar.max}
            onResize={(w) => patch((p) => ({ ...p, sidebar: { ...p.sidebar, width: w } }))}
            onReset={() => patch((p) => ({ ...p, sidebar: { ...p.sidebar, width: LIMITS.sidebar.def } }))}
          />
        )}
        <main className="flex min-h-0 min-w-0 flex-1 flex-col">{body}</main>
        {panel.open ? (
          <Splitter
            mode="grow-left"
            label="Redimensionar painel de modelos e limiares"
            className="max-lg:hidden"
            width={panel.width}
            min={LIMITS.panel.min}
            max={LIMITS.panel.max}
            onResize={(w) => patch((p) => ({ ...p, panel: { ...p.panel, width: w } }))}
            onReset={() => patch((p) => ({ ...p, panel: { ...p.panel, width: LIMITS.panel.def } }))}
          />
        ) : (
          <button
            type="button"
            onClick={() => setPanel(true)}
            className="flex w-6 shrink-0 flex-col items-center justify-center gap-2 border-l bg-card/40 text-[10px] font-medium text-muted-foreground hover:bg-accent hover:text-foreground focus-visible:bg-accent"
            aria-label="Abrir painel de modelos e limiares (P)"
            data-testid="edge-tab-processing"
          >
            <PanelRightOpen className="size-3.5" aria-hidden />
            <span className="[writing-mode:vertical-rl]">Modelos e limiares</span>
          </button>
        )}
        <ProcessingPanel
          runtime={runtime}
          open={panel.open}
          width={panel.width}
          tab={panel.tab}
          onTab={(t) => patch((p) => ({ ...p, panel: { ...p.panel, tab: t } }))}
          onClose={() => setPanel(false)}
        />
      </div>

      <AlertDialog open={confirmClear} onOpenChange={setConfirmClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Limpar a seleção?</AlertDialogTitle>
            <AlertDialogDescription>
              As {selected.length} câmeras deixarão de ser exibidas no grid. O processamento no backend <b>não</b> é interrompido e você pode selecioná-las de novo a qualquer momento.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancelar</AlertDialogCancel>
            <AlertDialogAction onClick={() => { clear(); setFocusId(null) }} data-testid="confirm-clear">Limpar seleção</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
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
