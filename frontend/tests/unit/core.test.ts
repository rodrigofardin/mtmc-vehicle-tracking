import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { capacity, columnsFor } from '@/features/monitoring/layouts'
import { drawOverlay, labelFor } from '@/features/tracking/drawOverlay'
import { TrailStore } from '@/features/tracking/trails'
import { parsePacket } from '@/services/streaming/protocol'
import { CameraStream } from '@/services/streaming/CameraStream'
import { StreamManager } from '@/services/streaming/StreamManager'
import { api, ApiError } from '@/services/api/client'
import type { Detection, FrameHeader } from '@/types/api'

const det = (id: number, x = 100, extra: Partial<Detection> = {}): Detection => ({
  id, cls: 2, name: 'car', conf: 0.87, bbox: [x, 50, x + 40, 90], gid: null, uncertain: false, ...extra,
})
const hdr = (frame: number, dets: Detection[], loop = 0): FrameHeader => ({
  cam: 'c001', seq: frame, frame, loop, t_global: frame / 10, t_local: frame / 10, width: 1000, height: 500,
  stream_width: 500, stream_height: 250, proc_fps: 10, inference_ms: 12, mode: 'live', detections: dets,
})

describe('layouts', () => {
  it('capacidade por layout e auto', () => {
    expect(capacity('4', 10)).toBe(4)
    expect(capacity('9', 3)).toBe(3)
    expect(capacity('auto', 13)).toBe(13)
  })
  it.each([
    ['1', 5, 1600, 1], ['2', 5, 1600, 2], ['4', 4, 1600, 2], ['6', 6, 1600, 3], ['9', 9, 1600, 3],
    ['auto', 1, 1600, 1], ['auto', 5, 1600, 3], ['auto', 16, 1600, 4], ['auto', 30, 1600, 5],
  ] as const)('layout %s com %i câmeras em %ipx => %i colunas', (l, n, w, cols) => {
    expect(columnsFor(l, n, w)).toBe(cols)
  })
  it('reduz colunas em janelas estreitas', () => {
    expect(columnsFor('9', 9, 600)).toBe(2)
    expect(columnsFor('auto', 9, 300)).toBe(1)
  })
})

describe('TrailStore', () => {
  it('acumula a base da caixa por ID e limita o histórico', () => {
    const t = new TrailStore(5, 30)
    for (let f = 0; f < 12; f++) t.update(hdr(f, [det(7, 100 + f * 10)]))
    const pts = t.get(7)
    expect(pts).toHaveLength(5)
    expect(pts[4]).toEqual({ x: 100 + 11 * 10 + 20, y: 90 })
  })
  it('mantém trilhas separadas por ID e descarta as ausentes', () => {
    const t = new TrailStore(10, 3)
    t.update(hdr(0, [det(1), det(2)]))
    t.update(hdr(1, [det(1)]))
    t.update(hdr(6, [det(1)]))
    expect(t.get(2)).toHaveLength(0)
    expect(t.get(1).length).toBeGreaterThan(0)
  })
  it('zera ao reiniciar o vídeo (novo loop ou frame voltando)', () => {
    const t = new TrailStore()
    t.update(hdr(50, [det(1)]))
    t.update(hdr(51, [det(1)]))
    t.update(hdr(2, [det(1)]))
    expect(t.get(1)).toHaveLength(1)
    t.update(hdr(3, [det(1)], 1))
    expect(t.get(1)).toHaveLength(1)
  })
})

describe('drawOverlay', () => {
  const mkCtx = () => {
    const calls: string[] = []
    const ctx = new Proxy({ measureText: () => ({ width: 40 }) } as Record<string, unknown>, {
      get: (o, k: string) => (k in o ? o[k] : (...a: unknown[]) => void calls.push(`${k}:${a.join(',')}`)),
      set: () => true,
    })
    return { ctx: ctx as unknown as CanvasRenderingContext2D, calls }
  }
  it('desenha caixa em escala, rótulo e trilha', () => {
    const t = new TrailStore()
    t.update(hdr(0, [det(1, 100)]))
    t.update(hdr(1, [det(1, 120)]))
    const { ctx, calls } = mkCtx()
    drawOverlay(ctx, hdr(1, [det(1, 120)]), t, { boxes: true, labels: true, trails: true }, 500)
    // bbox [120,50,160,90] no vídeo de 1000 px, canvas de 500 px (k = 0,5) => x=60, y=25, 20x20
    expect(calls).toContain('strokeRect:60,25,20,20')
    expect(calls.some((c) => c.startsWith('fillText:#1 car 87%'))).toBe(true)
    expect(calls.filter((c) => c.startsWith('lineTo')).length).toBeGreaterThan(0) // trilha
  })
  it('respeita os interruptores', () => {
    const t = new TrailStore()
    t.update(hdr(0, [det(1)]))
    t.update(hdr(1, [det(1, 120)]))
    const { ctx, calls } = mkCtx()
    drawOverlay(ctx, hdr(1, [det(1, 120)]), t, { boxes: false, labels: false, trails: false }, 1000)
    expect(calls.filter((c) => /strokeRect|fillText|lineTo/.test(c))).toHaveLength(0)
  })
  it('rótulo traz ID do tracker, classe, confiança e ID global quando existe', () => {
    expect(labelFor(det(12))).toBe('#12 car 87%')
    expect(labelFor(det(12, 0, { gid: 'G0004', uncertain: true }))).toBe('#12 car 87% G0004 ?')
  })
})

function pkt(h: FrameHeader, jpeg = [0xff, 0xd8, 1, 2]): ArrayBuffer {
  const hb = new TextEncoder().encode(JSON.stringify(h))
  const out = new Uint8Array(4 + hb.length + jpeg.length)
  new DataView(out.buffer).setUint32(0, hb.length, false)
  out.set(hb, 4)
  out.set(jpeg, 4 + hb.length)
  return out.buffer
}

describe('protocol', () => {
  it('decodifica header JSON e bytes JPEG', async () => {
    const p = parsePacket(pkt(hdr(3, [det(9)])))
    expect(p.header.detections[0].id).toBe(9)
    expect(p.jpeg.size).toBe(4)
    expect(p.jpeg.type).toBe('image/jpeg')
  })
  it('rejeita pacotes inválidos', () => {
    expect(() => parsePacket(new ArrayBuffer(2))).toThrow()
    const bad = new Uint8Array(10)
    new DataView(bad.buffer).setUint32(0, 999, false)
    expect(() => parsePacket(bad.buffer)).toThrow()
  })
})

class FakeSocket {
  static all: FakeSocket[] = []
  binaryType = ''
  onopen: (() => void) | null = null
  onmessage: ((e: { data: unknown }) => void) | null = null
  onclose: (() => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  url: string
  constructor(url: string) {
    this.url = url
    FakeSocket.all.push(this)
  }
  close() {
    this.closed = true
  }
  open() { this.onopen?.() }
  send(buf: ArrayBuffer) { this.onmessage?.({ data: buf }) }
  drop() { this.onclose?.() }
}
const mk = (url: string) => new FakeSocket(url) as unknown as WebSocket

describe('CameraStream / StreamManager', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    FakeSocket.all = []
  })
  afterEach(() => vi.useRealTimers())

  it('conecta, recebe metadados e reconecta com backoff após queda', () => {
    const s = new CameraStream('c001', { createSocket: mk, backoffBaseMs: 100, stalledMs: 10_000 })
    const states: string[] = []
    const got: number[] = []
    s.onState((st) => states.push(st))
    s.onPacket((p) => got.push(p.header.detections.length))
    expect(FakeSocket.all[0].url).toMatch(/^wss?:\/\/.*\/api\/cameras\/c001\/ws$/)
    FakeSocket.all[0].open()
    FakeSocket.all[0].send(pkt(hdr(1, [det(1), det(2)])))
    expect(got).toEqual([2])
    expect(s.currentState).toBe('streaming')

    FakeSocket.all[0].drop()
    expect(s.currentState).toBe('reconnecting')
    expect(FakeSocket.all).toHaveLength(1)
    vi.advanceTimersByTime(100)
    expect(FakeSocket.all).toHaveLength(2) // 1ª tentativa após backoff de 100 ms
    FakeSocket.all[1].drop() // falha de novo => backoff dobra (200 ms)
    vi.advanceTimersByTime(150)
    expect(FakeSocket.all).toHaveLength(2)
    vi.advanceTimersByTime(60)
    expect(FakeSocket.all).toHaveLength(3)
    FakeSocket.all[2].open()
    FakeSocket.all[2].send(pkt(hdr(2, [])))
    expect(s.currentState).toBe('streaming')
    expect(s.attempts).toBe(0)
    expect(states).toContain('reconnecting')
    s.dispose()
  })

  it('detecta fluxo parado (timeout) e reconecta', () => {
    const s = new CameraStream('c001', { createSocket: mk, backoffBaseMs: 50, stalledMs: 1000 })
    FakeSocket.all[0].open()
    vi.advanceTimersByTime(1500)
    expect(s.currentState).toBe('reconnecting')
    vi.advanceTimersByTime(100)
    expect(FakeSocket.all.length).toBeGreaterThan(1)
    s.dispose()
  })

  it('dispose fecha o socket e cancela novas tentativas', () => {
    const s = new CameraStream('c001', { createSocket: mk, backoffBaseMs: 50 })
    const ws = FakeSocket.all[0]
    s.dispose()
    expect(ws.closed).toBe(true)
    vi.advanceTimersByTime(60_000)
    expect(FakeSocket.all).toHaveLength(1)
  })

  it('compartilha uma conexão por câmera e só fecha após a carência', () => {
    const m = new StreamManager({ createSocket: mk }, 500)
    const a = m.acquire('c001')
    const b = m.acquire('c001')
    expect(a.stream).toBe(b.stream)
    expect(FakeSocket.all).toHaveLength(1)
    a.release()
    vi.advanceTimersByTime(1000)
    expect(m.openCount).toBe(1) // b ainda usa
    b.release()
    b.release() // idempotente
    vi.advanceTimersByTime(499)
    expect(m.openCount).toBe(1)
    vi.advanceTimersByTime(2)
    expect(m.openCount).toBe(0)
    expect(FakeSocket.all[0].closed).toBe(true)
  })

  it('remontar dentro da carência reaproveita a conexão (sem reiniciar o stream)', () => {
    const m = new StreamManager({ createSocket: mk }, 500)
    m.acquire('c001').release()
    vi.advanceTimersByTime(200)
    m.acquire('c001')
    vi.advanceTimersByTime(2000)
    expect(FakeSocket.all).toHaveLength(1)
    expect(m.openCount).toBe(1)
  })
})

describe('api client', () => {
  afterEach(() => vi.unstubAllGlobals())
  it('lista câmeras', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [{ id: 'c001' }] }))
    expect(await api.cameras()).toEqual([{ id: 'c001' }])
  })
  it('HTTP de erro => ApiError com status', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }))
    await expect(api.cameras()).rejects.toMatchObject({ name: 'ApiError', status: 503 })
  })
  it('backend inacessível => ApiError', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed')))
    await expect(api.health()).rejects.toBeInstanceOf(ApiError)
  })
})

