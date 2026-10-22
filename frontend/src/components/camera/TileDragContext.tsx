import { createContext, useContext } from 'react'
import { GripVertical } from 'lucide-react'
import { cn } from '@/lib/utils'

/** Ponte entre o item arrastável do grid (dnd-kit) e a "alça" desenhada dentro do painel da câmera. */
export interface TileDrag {
  setActivatorNodeRef: (el: HTMLElement | null) => void
  listeners?: Record<string, unknown>
  attributes?: Record<string, unknown>
  isDragging: boolean
  disabled: boolean
}

export const TileDragContext = createContext<TileDrag | null>(null)

/**
 * Alça de arraste. Só ela consome o contexto: durante o arraste apenas este componente re-renderiza
 * (o painel da câmera e o canvas de vídeo não são tocados).
 */
export function DragHandle({ label, className, children }: { label: string; className?: string; children: React.ReactNode }) {
  const d = useContext(TileDragContext)
  const active = !!d && !d.disabled
  return (
    <div
      ref={active ? d.setActivatorNodeRef : undefined}
      {...(active ? (d.listeners as object) : {})}
      {...(active ? (d.attributes as object) : {})}
      aria-label={active ? `Reordenar ${label}: arraste com o mouse, ou Espaço e setas no teclado` : undefined}
      data-testid="drag-handle"
      className={cn(
        'pointer-events-auto flex items-center gap-1.5 rounded bg-black/60 px-1.5 py-0.5 backdrop-blur-sm',
        active && 'cursor-grab touch-none active:cursor-grabbing',
        className,
      )}
    >
      {active && <GripVertical className="size-3.5 text-muted-foreground" aria-hidden />}
      {children}
    </div>
  )
}
