"""Construção de tracklets locais a partir das detecções por câmera."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .videos import VideoRegistry


@dataclass
class Tracklet:
    uid: str                      # "<cam>:<local_id>" — único entre câmeras
    cam: str
    local_id: int
    cls_name: str
    cls_conf: float               # fração de votos ponderados da classe dominante
    first_frame: int
    last_frame: int
    t_local_first: float
    t_local_last: float
    t_global_first: float
    t_global_last: float
    n_obs: int
    frames: List[int]
    boxes: List[List[float]]      # xyxy em pixels do vídeo original
    centers: List[List[float]]
    confs: List[float]
    mean_conf: float
    entry: str = "unknown"        # "border" | "inside" | nome de zona
    exit: str = "unknown"
    crops: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None   # preenchido apenas por um modelo Re-ID real
    source: str = ""              # CSV de detecções de origem (rastreabilidade)

    def _arrays(self):
        """(tempos globais por observação, caixas Nx4) em numpy; cache fora dos campos do dataclass."""
        c = self.__dict__.get("_arr_cache")
        if c is None or c[2] != len(self.frames):
            span = max(self.last_frame - self.first_frame, 1)
            fr = np.asarray(self.frames, dtype=float)
            tg = self.t_global_first + (fr - self.first_frame) * ((self.t_global_last - self.t_global_first) / span)
            c = (tg, np.asarray(self.boxes, dtype=float), len(self.frames))
            self.__dict__["_arr_cache"] = c
        return c[0], c[1]

    def center_at(self, t_global: float) -> Optional[List[float]]:
        """Centro interpolado no tempo global; None se fora do intervalo do tracklet."""
        if not (self.t_global_first - 1e-6 <= t_global <= self.t_global_last + 1e-6):
            return None
        tg, b = self._arrays()
        return [float(np.interp(t_global, tg, (b[:, 0] + b[:, 2]) / 2)),
                float(np.interp(t_global, tg, (b[:, 1] + b[:, 3]) / 2))]

    def foot_at(self, t_global: float) -> List[float]:
        """Ponto no chão (centro da base da caixa) interpolado no tempo global."""
        tg, b = self._arrays()
        return [float(np.interp(t_global, tg, (b[:, 0] + b[:, 2]) / 2)), float(np.interp(t_global, tg, b[:, 3]))]

    def feet_at(self, ts) -> np.ndarray:
        tg, b = self._arrays()
        return np.stack([np.interp(ts, tg, (b[:, 0] + b[:, 2]) / 2), np.interp(ts, tg, b[:, 3])], axis=1)

    def velocity(self) -> Optional[List[float]]:
        """Velocidade média em pixels/segundo (apenas no plano da imagem da própria câmera)."""
        dt = self.t_local_last - self.t_local_first
        if dt <= 0 or len(self.centers) < 2:
            return None
        a, b = np.array(self.centers[0]), np.array(self.centers[-1])
        return ((b - a) / dt).tolist()


def _point_zone(zones: Dict[str, list], pt) -> Optional[str]:
    for name, poly in zones.items():
        if cv2.pointPolygonTest(np.array(poly, dtype=np.float32), (float(pt[0]), float(pt[1])), False) >= 0:
            return name
    return None


def _classify_endpoint(box, size, margin, zones, center) -> str:
    z = _point_zone(zones, center) if zones else None
    if z:
        return z
    w, h = size
    mx, my = margin * w, margin * h
    x1, y1, x2, y2 = box
    if x1 <= mx or y1 <= my or x2 >= w - mx or y2 >= h - my:
        return "border"
    return "inside"


def build_tracklets(rows: List[dict], cam: str, size, cfg: dict, source: str = "") -> List[Tracklet]:
    groups = defaultdict(list)
    for r in rows:
        groups[r["track_id"]].append(r)
    tcfg = cfg["tracklets"]
    zones = (cfg.get("spatial", {}).get("zones") or {}).get(cam, {})
    out = []
    for tid, g in sorted(groups.items()):
        if len(g) < tcfg["min_length"]:
            continue
        g.sort(key=lambda r: r["frame"])
        votes = defaultdict(float)
        for r in g:
            votes[r["cls_name"]] += r["conf"]
        cls = max(votes, key=votes.get)
        boxes = [r["bbox"] for r in g]
        centers = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
        out.append(Tracklet(
            uid=f"{cam}:{tid}", cam=cam, local_id=tid, cls_name=cls,
            cls_conf=votes[cls] / sum(votes.values()),
            first_frame=g[0]["frame"], last_frame=g[-1]["frame"],
            t_local_first=g[0]["t_local"], t_local_last=g[-1]["t_local"],
            t_global_first=g[0]["t_global"], t_global_last=g[-1]["t_global"],
            n_obs=len(g), frames=[r["frame"] for r in g], boxes=boxes, centers=centers,
            confs=[r["conf"] for r in g], mean_conf=float(np.mean([r["conf"] for r in g])),
            entry=_classify_endpoint(boxes[0], size, tcfg["border_margin"], zones, centers[0]),
            exit=_classify_endpoint(boxes[-1], size, tcfg["border_margin"], zones, centers[-1]),
            source=source))
    return out


def save_tracklets(tracklets: List[Tracklet], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(t) for t in tracklets]), encoding="utf-8")


def load_tracklets(path) -> List[Tracklet]:
    return [Tracklet(**d) for d in json.loads(Path(path).read_text(encoding="utf-8"))]


def extract_crops(tracklets: List[Tracklet], registry: VideoRegistry, cfg: dict) -> None:
    """Salva até K recortes por tracklet (maior conf*área) lendo o vídeo sequencialmente."""
    k = cfg["tracklets"]["crops_per_tracklet"]
    if k <= 0:
        return
    out_root = Path(cfg["output"]["dir"]) / cfg["tracklets"]["crop_dir"]
    by_cam = defaultdict(list)
    for t in tracklets:
        by_cam[t.cam].append(t)
    for cam, ts in by_cam.items():
        info = registry[cam]
        if not info.accessible:
            continue
        want = defaultdict(list)   # frame -> [(tracklet, idx)]
        for t in ts:
            score = [c * (b[2] - b[0]) * (b[3] - b[1]) for c, b in zip(t.confs, t.boxes)]
            for i in sorted(np.argsort(score)[::-1][:k]):
                want[t.frames[int(i)]].append((t, int(i)))
        if not want:
            continue
        cap = cv2.VideoCapture(info.path)
        (out_root / cam).mkdir(parents=True, exist_ok=True)
        last = max(want)
        for fi in range(last + 1):
            if fi not in want:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            for t, i in want[fi]:
                x1, y1, x2, y2 = (int(round(v)) for v in t.boxes[i])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, img.shape[1]), min(y2, img.shape[0])
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                p = out_root / cam / f"{t.local_id}_{fi}.jpg"
                cv2.imwrite(str(p), img[y1:y2, x1:x2])
                t.crops.append(str(p))
        cap.release()
