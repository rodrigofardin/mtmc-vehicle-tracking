import { useMemo, useRef } from 'react'
import { CameraPanel } from '@/components/camera/CameraPanel'
import type { OverlayOptions } from '@/features/tracking/drawOverlay'
import { capacity, columnsFor } from '@/features/monitoring/layouts'
import { useElementWidth } from '@/hooks/useElementWidth'
import type { CameraStatus, LayoutId } from '@/types/api'

interface Props {
  cameras: CameraStatus[] // já filtradas pela seleção
  layout: LayoutId
  overlay: OverlayOptions
  focusId: string | null
  onFocus: (id: string | null) => void
  onRemove: (id: string) => void
}

export function VideoGrid({ cameras, layout, overlay, focusId, onFocus, onRemove }: Props) {
  const ref = useRef<HTMLDivElement>(null)
  const width = useElementWidth(ref)
  const focused = focusId ? cameras.find((c) => c.id === focusId) : undefined

  const shown = useMemo(
    () => (focused ? [focused] : cameras.slice(0, capacity(layout, cameras.length))),
    [focused, cameras, layout],
  )
  const cols = focused ? 1 : columnsFor(layout, shown.length, width)
  const rows = Math.ceil(shown.length / cols)
  const hidden = cameras.length - shown.length

  return (
    <div ref={ref} className="flex min-h-0 flex-1 flex-col gap-2 p-2" data-testid="video-grid" data-cols={cols} data-count={shown.length}>
      {hidden > 0 && !focused && (
        <p className="rounded border border-warn/40 bg-warn/10 px-3 py-1 text-xs text-warn" role="note">
          Exibindo {shown.length} de {cameras.length} câmeras selecionadas neste layout. As demais continuam selecionadas.
        </p>
      )}
      <div
        className="grid min-h-0 flex-1 gap-2"
        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`, gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))` }}
      >
        {shown.map((c) => (
          <CameraPanel key={c.id} camera={c} overlay={overlay} expanded={!!focused} onExpand={onFocus} onRemove={onRemove} />
        ))}
      </div>
    </div>
  )
}
