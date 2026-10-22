"""Vídeos de diagnóstico: por câmera e mosaico sincronizado (grade dinâmica)."""
from __future__ import annotations

import colorsys
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .detect_track import read_detections, det_paths
from .tracklets import Tracklet
from .videos import VideoRegistry

log = logging.getLogger(__name__)


def color_for(key) -> tuple:
    h = (hash(str(key)) * 2654435761 % 1000) / 1000.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def build_lookup(cfg, cams, tracklets: List[Tracklet], gid_map: Optional[Dict[str, str]],
                 uncertain_uids: set):
    det = {}
    for c in cams:
        p, _ = det_paths(cfg, c)
        by_frame = defaultdict(list)
        if p.is_file():
            for r in read_detections(p):
                r["uid"] = f"{c}:{r['track_id']}"
                r["gid"] = (gid_map or {}).get(r["uid"])
                r["uncertain"] = r["uid"] in uncertain_uids
                by_frame[r["frame"]].append(r)
        det[c] = by_frame
    return det


def draw(img, rows, t_global, mode, cam, trails=None):
    for r in rows:
        x1, y1, x2, y2 = (int(v) for v in r["bbox"])
        key = r["gid"] if (mode != "local" and r["gid"]) else r["uid"]
        col = color_for(key)
        unc = r.get("uncertain")
        thick = 2
        if unc:   # associação incerta: tracejado simples
            for x in range(x1, x2, 12):
                cv2.line(img, (x, y1), (min(x + 6, x2), y1), col, thick)
                cv2.line(img, (x, y2), (min(x + 6, x2), y2), col, thick)
            for y in range(y1, y2, 12):
                cv2.line(img, (x1, y), (x1, min(y + 6, y2)), col, thick)
                cv2.line(img, (x2, y), (x2, min(y + 6, y2)), col, thick)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)
        parts = [f"L{r['track_id']}"]
        if mode in ("global", "compare") and r["gid"]:
            parts.append(r["gid"])
        parts.append(f"{r['cls_name']} {r['conf']:.2f}")
        if unc:
            parts.append("?")
        label = " ".join(parts)
        s = max(img.shape[1] / 1920, 0.5)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s * 1.5, 2)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s * 1.5,
                    (0, 0, 0), 2, cv2.LINE_AA)
    s = max(img.shape[1] / 1920, 0.5)
    cv2.putText(img, f"{cam}  t_global={t_global:8.2f}s  [{mode}]", (10, int(40 * s * 1.3)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s * 1.3, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, f"{cam}  t_global={t_global:8.2f}s  [{mode}]", (10, int(40 * s * 1.3)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s * 1.3, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def render_camera(cfg, registry: VideoRegistry, cam, lookup, mode, out_path, max_seconds=None):
    info = registry[cam]
    cap = cv2.VideoCapture(info.path)
    w, h = info.width, info.height
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (w, h))
    n = info.n_frames
    if max_seconds:
        n = min(n, int(max_seconds * info.fps))
    for i in range(n):
        ok, img = cap.read()
        if not ok:
            log.warning("%s: leitura falhou no frame %d; encerrando vídeo", cam, i)
            break
        vw.write(draw(img, lookup[cam].get(i, []), info.to_global(i), mode, cam))
    cap.release()
    vw.release()


def render_mosaic(cfg, registry: VideoRegistry, cams, lookup, mode, out_path, cell_w=640,
                  fps=None, max_seconds=None, start=None):
    """Mosaico alinhado pelo tempo GLOBAL (não pelo índice de frame)."""
    cams = [c for c in cams if registry[c].accessible]
    if not cams:
        raise RuntimeError("nenhuma câmera utilizável para o mosaico")
    n = len(cams)
    cols = math.ceil(math.sqrt(n))
    rows_n = math.ceil(n / cols)
    cell_h = int(cell_w * 9 / 16)
    fps = fps or min(registry[c].fps for c in cams)
    tl = registry.synchronized_overlap() or registry.common_timeline()
    t0 = tl[0] if start is None else start
    t1 = tl[1]
    if max_seconds:
        t1 = min(t1, t0 + max_seconds)
    caps = {c: cv2.VideoCapture(registry[c].path) for c in cams}
    pos = {c: -1 for c in cams}
    last = {c: None for c in cams}
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (cols * cell_w, rows_n * cell_h))
    steps = int((t1 - t0) * fps)
    for s in range(steps):
        t = t0 + s / fps
        canvas = np.zeros((rows_n * cell_h, cols * cell_w, 3), np.uint8)
        for k, c in enumerate(cams):
            info = registry[c]
            fi = info.to_local_frame(t)
            cell = np.zeros((cell_h, cell_w, 3), np.uint8)
            if 0 <= fi < info.n_frames:
                while pos[c] < fi:
                    ok = caps[c].grab()
                    pos[c] += 1
                    if not ok:
                        break
                    if pos[c] == fi:
                        ok, img = caps[c].retrieve()
                        if ok:
                            last[c] = (fi, img)
                if last[c] is not None and last[c][0] == fi:
                    img = draw(last[c][1].copy(), lookup[c].get(fi, []), t, mode, c)
                    cell = cv2.resize(img, (cell_w, cell_h))
                elif last[c] is not None:      # fps maior: repete o último frame lido
                    cell = cv2.resize(draw(last[c][1].copy(), lookup[c].get(last[c][0], []), t, mode, c),
                                      (cell_w, cell_h))
            else:
                cv2.putText(cell, f"{c}: sem video em t={t:.2f}s", (10, cell_h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            r, q = divmod(k, cols)
            canvas[r * cell_h:(r + 1) * cell_h, q * cell_w:(q + 1) * cell_w] = cell
        vw.write(canvas)
    for c in caps.values():
        c.release()
    vw.release()
    return steps
