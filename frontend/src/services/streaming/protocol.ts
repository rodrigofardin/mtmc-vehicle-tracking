import type { FrameHeader, FramePacket } from '@/types/api'

/** Pacote binário: [uint32 BE tamanho do header][header JSON UTF-8][bytes JPEG]. */
export function parsePacket(buf: ArrayBuffer): FramePacket {
  if (buf.byteLength < 6) throw new Error('pacote curto demais')
  const hlen = new DataView(buf).getUint32(0, false)
  if (4 + hlen >= buf.byteLength) throw new Error('header inconsistente')
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hlen))) as FrameHeader
  return { header, jpeg: new Blob([buf.slice(4 + hlen)], { type: 'image/jpeg' }) }
}
