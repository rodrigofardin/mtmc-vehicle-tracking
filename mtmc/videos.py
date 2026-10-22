"""Gerenciamento de vídeos e alinhamento temporal.

Convenções (documentadas também no README):
  * `t_local`  = índice_do_frame / fps do vídeo (tempo de mídia, começa em 0).
  * `t_global` = t_local + offset_câmera [+ deriva], com offset vindo de cam_timestamp
    **somente** quando `timestamp.semantics == "start_offset"`.
  * Semânticas suportadas: `start_offset` (offset em segundos do início do vídeo em
    relação a uma referência comum), `none` (ignora cam_timestamp: offset 0 para todas).
  * `cam_framenum` é interpretado como `total` de frames (convenção `count`) ou
    `last_index` (total = valor+1), configurável; a convenção é validada contra o vídeo.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from .metadata import MetadataError, read_framenums, read_timestamps

log = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    cam_id: str
    path: str
    accessible: bool = False
    fps: float = 0.0
    width: int = 0
    height: int = 0
    reported_frames: int = 0          # CAP_PROP_FRAME_COUNT
    decoded_frames: Optional[int] = None  # contagem real por decodificação (opcional)
    duration: float = 0.0             # s, a partir de reported_frames/fps
    meta_frames_raw: Optional[int] = None  # valor cru de cam_framenum
    meta_frames: Optional[int] = None      # total normalizado
    meta_timestamp_raw: Optional[float] = None
    offset: float = 0.0               # offset normalizado usado (s)
    drift_ppm: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """Número de frames usado no processamento (decodificado > reportado > metadado)."""
        if self.decoded_frames is not None:
            return self.decoded_frames
        return self.reported_frames

    def to_global(self, frame_idx: float) -> float:
        t_local = frame_idx / self.fps if self.fps > 0 else 0.0
        return t_local * (1.0 + self.drift_ppm * 1e-6) + self.offset

    def to_local_frame(self, t_global: float) -> int:
        t_local = (t_global - self.offset) / (1.0 + self.drift_ppm * 1e-6)
        return int(round(t_local * self.fps))

    @property
    def global_start(self) -> float:
        return self.to_global(0)

    @property
    def global_end(self) -> float:
        return self.to_global(max(self.n_frames - 1, 0))


def probe_video(cam_id: str, path, count_frames: bool = False) -> VideoInfo:
    info = VideoInfo(cam_id=cam_id, path=str(path))
    p = Path(path)
    if not p.is_file():
        info.warnings.append(f"arquivo de vídeo inexistente: {p}")
        return info
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            info.warnings.append(f"vídeo não pôde ser aberto: {p}")
            return info
        info.fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        info.reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if info.fps <= 0:
            info.warnings.append("FPS inválido reportado pelo vídeo")
        else:
            info.duration = info.reported_frames / info.fps
        info.accessible = info.fps > 0 and info.width > 0
        if count_frames and info.accessible:
            n = 0
            while cap.grab():
                n += 1
            info.decoded_frames = n
    finally:
        cap.release()
    return info


class VideoRegistry:
    """Conjunto dinâmico de câmeras. Nada aqui assume número fixo de câmeras."""

    def __init__(self, infos: Dict[str, VideoInfo], notes: Optional[List[str]] = None,
                 semantics: Dict | None = None):
        self.infos = infos
        self.notes = notes or []
        self.semantics = semantics or {}

    @property
    def cam_ids(self) -> List[str]:
        return list(self.infos)

    def __getitem__(self, cam_id: str) -> VideoInfo:
        return self.infos[cam_id]

    def usable(self) -> List[str]:
        return [c for c, i in self.infos.items() if i.accessible]

    def to_dict(self) -> dict:
        return {"semantics": self.semantics, "notes": self.notes,
                "cameras": {c: asdict(i) for c, i in self.infos.items()}}

    @classmethod
    def from_config(cls, cfg: dict, count_frames: bool = False) -> "VideoRegistry":
        meta = cfg.get("metadata", {})
        ts_cfg = cfg.get("timestamp", {})
        fn_cfg = cfg.get("framenum", {})
        sem = ts_cfg.get("semantics", "start_offset")
        if sem not in ("start_offset", "none"):
            raise MetadataError(f"timestamp.semantics inválido: {sem!r} (use start_offset|none)")
        conv = fn_cfg.get("convention", "count")
        if conv not in ("count", "last_index"):
            raise MetadataError(f"framenum.convention inválido: {conv!r}")

        notes: List[str] = []
        ts, fn = {}, {}
        for key, reader, out in (("timestamp_file", read_timestamps, ts),
                                 ("framenum_file", read_framenums, fn)):
            f = meta.get(key)
            if not f:
                notes.append(f"metadata.{key} não configurado")
                continue
            try:
                tab = reader(f)
                out.update(tab.values)
                notes.extend(tab.warnings)
            except MetadataError as e:
                notes.append(f"metadado inválido/ausente ({key}): {e}")
        if sem == "start_offset" and not ts:
            notes.append("semântica start_offset configurada mas sem timestamps: offsets = 0")

        drift = cfg.get("alignment", {}).get("drift_ppm", {}) or {}
        infos: Dict[str, VideoInfo] = {}
        cams = cfg.get("cameras") or []
        if not cams:
            raise MetadataError("lista 'cameras' vazia na configuração")
        for c in cams:
            cid, path = str(c["id"]), c["video"]
            if cid in infos:
                raise MetadataError(f"câmera duplicada na configuração: {cid}")
            info = probe_video(cid, path, count_frames=count_frames)
            if cid in fn:
                info.meta_frames_raw = int(fn[cid])
                info.meta_frames = info.meta_frames_raw + (1 if conv == "last_index" else 0)
            else:
                info.warnings.append("sem entrada em cam_framenum")
            if cid in ts:
                info.meta_timestamp_raw = float(ts[cid])
            else:
                info.warnings.append("sem entrada em cam_timestamp")
            if sem == "start_offset" and info.meta_timestamp_raw is not None:
                info.offset = info.meta_timestamp_raw + float(c.get("offset_adjust", 0.0))
            info.drift_ppm = float(drift.get(cid, 0.0))
            if info.accessible and info.meta_frames is not None:
                ref = info.n_frames
                if ref != info.meta_frames:
                    info.warnings.append(
                        f"cam_framenum ({info.meta_frames}) difere do vídeo ({ref})")
            infos[cid] = info
        # câmeras presentes nos metadados mas ausentes na configuração
        for cid in sorted((set(ts) | set(fn)) - set(infos)):
            notes.append(f"câmera {cid} existe nos metadados mas não na configuração (ignorada)")
        reg = cls(infos, notes, {"timestamp": sem, "framenum": conv,
                                 "timestamp_confirmed": bool(ts_cfg.get("confirmed", False))})
        if sem == "start_offset" and not reg.semantics["timestamp_confirmed"]:
            reg.notes.append(
                "INCERTEZA: semântica de cam_timestamp (início de vídeo em segundos, offset "
                "contra referência comum) não está documentada nos arquivos locais; adotada "
                "por consistência estrutural. Ajuste timestamp.semantics/confirmed no YAML.")
        return reg

    def common_timeline(self) -> Optional[tuple]:
        us = [self.infos[c] for c in self.usable()]
        if not us:
            return None
        return min(i.global_start for i in us), max(i.global_end for i in us)

    def synchronized_overlap(self) -> Optional[tuple]:
        """Intervalo global em que todas as câmeras utilizáveis têm vídeo."""
        us = [self.infos[c] for c in self.usable()]
        if not us:
            return None
        lo, hi = max(i.global_start for i in us), min(i.global_end for i in us)
        return (lo, hi) if hi > lo else None
