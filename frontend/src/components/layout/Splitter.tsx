import { useCallback, useRef } from 'react'
import { cn } from '@/lib/utils'

interface Props {
  /** `grow-right`: arrastar para a direita aumenta o painel à esquerda; `grow-left`: arrastar para a esquerda aumenta o painel à direita. */
  mode: 'grow-right' | 'grow-left'
  width: number
  min: number
  max: number
  onResize: (w: number) => void
  onReset?: () => void
  label: string
  className?: string
}

/** Divisor redimensionável (mouse/toque/teclado): setas ±16 px (Shift ±64), Home/End = mín/máx, duplo clique = padrão. */
export function Splitter({ mode, width, min, max, onResize, onReset, label, className }: Props) {
  const drag = useRef<{ x: number; w: number } | null>(null)
  const raf = useRef(0)
  const clamp = useCallback((w: number) => Math.min(max, Math.max(min, Math.round(w))), [min, max])

  const onPointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drag.current = { x: e.clientX, w: width }
    document.body.classList.add('is-resizing')
  }
  const onPointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    const d = drag.current
    if (!d) return
    const delta = (e.clientX - d.x) * (mode === 'grow-right' ? 1 : -1)
    cancelAnimationFrame(raf.current)
    raf.current = requestAnimationFrame(() => onResize(clamp(d.w + delta)))
  }
  const end = (e: React.PointerEvent<HTMLDivElement>) => {
    if (!drag.current) return
    drag.current = null
    document.body.classList.remove('is-resizing')
    if (e.currentTarget.hasPointerCapture(e.pointerId)) e.currentTarget.releasePointerCapture(e.pointerId)
  }
  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 64 : 16
    const sign = mode === 'grow-right' ? 1 : -1
    if (e.key === 'ArrowRight') onResize(clamp(width + step * sign))
    else if (e.key === 'ArrowLeft') onResize(clamp(width - step * sign))
    else if (e.key === 'Home') onResize(min)
    else if (e.key === 'End') onResize(max)
    else if (e.key === 'Enter') onReset?.()
    else return
    e.preventDefault()
  }

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={width}
      tabIndex={0}
      data-testid={`splitter-${mode}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onKeyDown={onKeyDown}
      onDoubleClick={onReset}
      className={cn(
        'group relative z-10 w-1.5 shrink-0 cursor-col-resize touch-none select-none bg-transparent outline-none',
        'before:absolute before:inset-y-0 before:left-1/2 before:w-px before:-translate-x-1/2 before:bg-border before:transition-colors',
        'hover:before:w-0.5 hover:before:bg-primary/70 focus-visible:before:w-0.5 focus-visible:before:bg-primary',
        className,
      )}
      title="Arraste para redimensionar (duplo clique restaura)"
    />
  )
}
