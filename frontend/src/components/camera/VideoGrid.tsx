import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  DndContext,
  KeyboardSensor,
  PointerSensor,
  pointerWithin,
  rectIntersection,
  useDraggable,
  useDroppable,
  useSensor,
  useSensors,
  type CollisionDetection,
  type DragEndEvent,
  type DragStartEvent,
  type PointerSensorOptions,
} from '@dnd-kit/core'
import { CameraPanel } from '@/components/camera/CameraPanel'
import { TileDragContext } from '@/components/camera/TileDragContext'
import { bestGrid, capacity } from '@/features/monitoring/layouts'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import { useElementSize } from '@/hooks/useElementSize'
import { cn } from '@/lib/utils'
import type { CameraStatus, LayoutId } from '@/types/api'

/** Pointer sensor que NÃO inicia arraste ao clicar em botões/links/campos nem em elementos `data-no-dnd`. */
class SmartPointerSensor extends PointerSensor {
  static activators = [
    {
      eventName: 'onPointerDown' as const,
      handler: ({ nativeEvent: e }: { nativeEvent: PointerEvent }, { onActivation }: PointerSensorOptions) => {
        if (!e.isPrimary || e.button !== 0) return false
        if ((e.target as Element | null)?.closest('button, a, input, select, textarea, [data-no-dnd]')) return false
        onActivation?.({ event: e })
        return true
      },
    },
  ]
}

/** Máximo de painéis montados ao mesmo tempo (visíveis + ocultos com conexão mantida). */
export const MAX_MOUNTED = 16

const collision: CollisionDetection = (args) => {
  const hits = pointerWithin(args)
  return hits.length ? hits : rectIntersection(args)
}

const PT = {
  screenReaderInstructions: {
    draggable: 'Para reordenar, pressione Espaço, mova com as setas até outra câmera e pressione Espaço para trocar; Esc cancela.',
  },
  announcements: {
    onDragStart: ({ active }: { active: { id: string | number } }) => `Câmera ${active.id} selecionada para reordenar.`,
    onDragOver: ({ active, over }: { active: { id: string | number }; over: { id: string | number } | null }) =>
      over ? `Câmera ${active.id} sobre a posição de ${over.id}.` : `Câmera ${active.id} fora de qualquer posição.`,
    onDragEnd: ({ active, over }: { active: { id: string | number }; over: { id: string | number } | null }) =>
      over && over.id !== active.id ? `Câmeras ${active.id} e ${over.id} trocaram de posição.` : `Câmera ${active.id} solta sem mudança.`,
    onDragCancel: ({ active }: { active: { id: string | number } }) => `Reordenação de ${active.id} cancelada.`,
  },
}

interface TileProps {
  id: string
  hidden: boolean
  dragDisabled: boolean
  flash: boolean
  children: React.ReactNode
}

/** Envolve um painel como item arrastável E alvo de soltura, sem remontar o painel (só muda `transform`/classes). */
function GridTile({ id, hidden, dragDisabled, flash, children }: TileProps) {
  const drag = useDraggable({ id, disabled: dragDisabled })
  const drop = useDroppable({ id, disabled: dragDisabled })
  const setRef = useCallback(
    (el: HTMLElement | null) => {
      drag.setNodeRef(el)
      drop.setNodeRef(el)
    },
    [drag.setNodeRef, drop.setNodeRef], // eslint-disable-line react-hooks/exhaustive-deps
  )
  const t = drag.transform
  const target = drop.isOver && !drag.isDragging
  const ctx = useMemo(
    () => ({
      setActivatorNodeRef: drag.setActivatorNodeRef,
      listeners: drag.listeners as Record<string, unknown> | undefined,
      attributes: drag.attributes as unknown as Record<string, unknown>,
      isDragging: drag.isDragging,
      disabled: dragDisabled,
    }),
    [drag.setActivatorNodeRef, drag.listeners, drag.attributes, drag.isDragging, dragDisabled],
  )
  return (
    <div
      ref={setRef}
      data-testid={`tile-${id}`}
      data-dragging={drag.isDragging || undefined}
      data-drop-target={target || undefined}
      style={t ? { transform: `translate3d(${Math.round(t.x)}px, ${Math.round(t.y)}px, 0)` } : undefined}
      className={cn(
        'relative min-h-0 min-w-0 rounded-md',
        hidden && 'hidden',
        drag.isDragging && 'z-40 opacity-90 shadow-2xl ring-2 ring-primary',
        target && 'ring-2 ring-primary ring-offset-2 ring-offset-background',
        flash && !drag.isDragging && 'ring-2 ring-primary/70',
      )}
    >
      <TileDragContext.Provider value={ctx}>{children}</TileDragContext.Provider>
      {target && (
        <div className="pointer-events-none absolute inset-0 z-30 flex items-center justify-center rounded-md bg-primary/15" aria-hidden>
          <span className="rounded bg-primary px-2 py-1 text-xs font-semibold text-primary-foreground shadow">Trocar com {id}</span>
        </div>
      )}
    </div>
  )
}

interface Props {
  cameras: CameraStatus[] // já filtradas pela seleção e na ordem preferida
  layout: LayoutId
  overlayFor: (id: string) => OverlayOptions
  focusId: string | null
  onFocus: (id: string | null) => void
  onRemove: (id: string) => void
  onOverlayChange: (id: string, patch: Partial<OverlayOptions>) => void
  onInfo: (id: string) => void
  onSwap: (a: string, b: string) => void
}

export function VideoGrid({ cameras, layout, overlayFor, focusId, onFocus, onRemove, onOverlayChange, onInfo, onSwap }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const { width, height } = useElementSize(ref)
  const [flash, setFlash] = useState<string[]>([])
  const flashTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(() => () => void (flashTimer.current && clearTimeout(flashTimer.current)), [])

  const focused = focusId ? cameras.find((c) => c.id === focusId) : undefined
  const cap = capacity(layout, cameras.length)
  const visibleIds = useMemo(() => new Set(focused ? [focused.id] : cameras.slice(0, cap).map((c) => c.id)), [focused, cameras, cap])
  // Tiles montados = visíveis + (até MAX_MOUNTED) ocultos. Ocultos mantêm a conexão mas não decodificam/desenham:
  // maximizar ou trocar de layout não reinicia streams. Acima do limite, os excedentes são desmontados.
  const mounted = useMemo(() => {
    const keep = new Set(visibleIds)
    for (const c of cameras) if (keep.size < MAX_MOUNTED) keep.add(c.id)
    return cameras.filter((c) => keep.has(c.id))
  }, [cameras, visibleIds])
  const hiddenByCap = cameras.length - visibleIds.size
  const shownCount = visibleIds.size
  const spec = focused ? { cols: 1, rows: 1 } : bestGrid(shownCount, width, Math.max(height - (hiddenByCap > 0 ? 28 : 0), 1))

  const sensors = useSensors(
    useSensor(SmartPointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor),
  )
  const [dragId, setDragId] = useState<string | null>(null)
  const onDragStart = (e: DragStartEvent) => setDragId(String(e.active.id))
  const onDragEnd = (e: DragEndEvent) => {
    setDragId(null)
    const a = String(e.active.id)
    const b = e.over ? String(e.over.id) : null
    if (!b || a === b) return
    onSwap(a, b)
    setFlash([a, b])
    if (flashTimer.current) clearTimeout(flashTimer.current)
    flashTimer.current = setTimeout(() => setFlash([]), 900)
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-1.5 p-1.5" data-testid="video-grid" data-cols={spec.cols} data-count={shownCount} data-dragging={dragId ?? undefined}>
      {hiddenByCap > 0 && !focused && (
        <p className="rounded border border-warn/40 bg-warn/10 px-3 py-0.5 text-[11px] text-warn" role="note">
          Exibindo {shownCount} de {cameras.length} câmeras selecionadas neste layout. As demais continuam selecionadas.
        </p>
      )}
      <DndContext
        sensors={sensors}
        collisionDetection={collision}
        onDragStart={onDragStart}
        onDragEnd={onDragEnd}
        onDragCancel={() => setDragId(null)}
        accessibility={PT as never}
      >
        <div
          ref={ref}
          className="grid min-h-0 flex-1 gap-1.5"
          style={{ gridTemplateColumns: `repeat(${spec.cols}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${spec.rows}, minmax(0, 1fr))` }}
        >
          {mounted.map((c) => (
            <GridTile key={c.id} id={c.id} hidden={!visibleIds.has(c.id)} dragDisabled={!!focused || shownCount < 2 || !visibleIds.has(c.id)} flash={flash.includes(c.id)}>
              <CameraPanel
                camera={c}
                overlay={overlayFor(c.id)}
                expanded={focusId === c.id}
                hidden={!visibleIds.has(c.id)}
                onExpand={onFocus}
                onRemove={onRemove}
                onOverlayChange={onOverlayChange}
                onInfo={onInfo}
              />
            </GridTile>
          ))}
        </div>
      </DndContext>
    </div>
  )
}
