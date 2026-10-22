"""Detecção (YOLO) + tracking local (ByteTrack) independentes por câmera."""
from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml

from .videos import VideoInfo

log = logging.getLogger(__name__)

FIELDS = ["cam", "frame", "t_local", "t_global", "track_id", "cls_id", "cls_name",
          "conf", "x1", "y1", "x2", "y2"]


def resolve_classes(spec, names: Dict[int, str]) -> Optional[List[int]]:
    """Converte nomes/índices de configuração em índices, lendo os nomes do modelo."""
    if spec is None:
        return None
    inv = {v: k for k, v in names.items()}
    out = []
    for s in spec:
        if isinstance(s, int):
            if s not in names:
                raise ValueError(f"classe {s} inexistente no modelo")
            out.append(s)
        else:
            if s not in inv:
                raise ValueError(f"classe {s!r} inexistente no modelo; disponíveis: {sorted(inv)}")
            out.append(inv[s])
    return out


def _params_key(cfg: dict, info: VideoInfo) -> str:
    p = {"det": {k: v for k, v in cfg["detector"].items()}, "trk": cfg["tracker"],
         "proc": {k: cfg["processing"][k] for k in ("start_frame", "end_frame", "stride")},
         "video": info.path, "n": info.n_frames}
    return hashlib.sha1(json.dumps(p, sort_keys=True, default=str).encode()).hexdigest()[:16]


def det_paths(cfg: dict, cam_id: str):
    d = Path(cfg["output"]["dir"]) / "detections"
    return d / f"{cam_id}.csv", d / f"{cam_id}.meta.json"


def _write_tracker_yaml(cfg: dict, out_dir: Path) -> str:
    t = dict(cfg["tracker"])
    t["tracker_type"] = "bytetrack"
    p = out_dir / "bytetrack.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(t), encoding="utf-8")
    return str(p)


class _LegacyStepper:
    """YOLO + `model.track` do Ultralytics (comportamento original; `detector.type` ausente ou `ultralytics_track`)."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO  # import tardio: testes sintéticos não exigem torch
        dcfg = cfg["detector"]
        if dcfg.get("disable_mkldnn", True) and str(dcfg["device"]).startswith("cpu"):
            import torch
            torch.backends.mkldnn.enabled = False
        self.d = dcfg
        self.model = YOLO(dcfg["model"])
        self.names = dict(self.model.names)
        self.classes = resolve_classes(dcfg.get("classes"), self.names)
        self.tracker_yaml = _write_tracker_yaml(cfg, Path(cfg["output"]["dir"]))

    def track(self, frame):
        d = self.d
        res = self.model.track(frame, persist=True, tracker=self.tracker_yaml, conf=d["conf"], iou=d["iou"],
                               imgsz=d["imgsz"], device=d["device"], classes=self.classes,
                               half=d.get("half", False), verbose=False)[0]
        b = res.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), None
        ids = b.id.cpu().numpy().astype(int) if b.id is not None else None
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), ids


class _UnifiedStepper:
    """Qualquer `Detector` (PyTorch/TensorRT/RF-DETR/D-FINE) + BYTETracker independente."""

    def __init__(self, cfg: dict, fps: float):
        from .detectors import COCO80, build_from_config
        from .local_tracker import LocalTracker
        d = cfg["detector"]
        self.det = build_from_config(d, Path(d.get("models_dir", Path(d["model"]).parent if d.get("model") else "models")))
        self.tracker = LocalTracker(cfg["tracker"], fps)
        self.conf = float(d["conf"])
        self.names = dict(COCO80)
        wanted = d.get("classes")
        inv = {v: k for k, v in COCO80.items()}
        self.allowed = None if wanted is None else {inv[c] if isinstance(c, str) else int(c) for c in wanted}

    def track(self, frame):
        xyxy, conf, cls = self.det.detect(frame, self.conf)
        if self.allowed is not None and len(cls):
            m = np.isin(cls, list(self.allowed))
            xyxy, conf, cls = xyxy[m], conf[m], cls[m]
        tr = self.tracker.update(xyxy, conf, cls, frame)
        if not tr:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int), np.zeros(0, int)
        return (np.array([t["bbox"] for t in tr]), np.array([t["conf"] for t in tr]),
                np.array([t["cls"] for t in tr], int), np.array([t["id"] for t in tr], int))


def make_stepper(cfg: dict, info: VideoInfo):
    if cfg["detector"].get("type", "ultralytics_track") == "ultralytics_track":
        return _LegacyStepper(cfg)
    return _UnifiedStepper(cfg, info.fps)


def run_camera(cfg: dict, info: VideoInfo, force: bool = False) -> dict:
    """Roda YOLO+ByteTrack em UMA câmera, com modelo/tracker próprios. Retorna estatísticas."""
    csv_p, meta_p = det_paths(cfg, info.cam_id)
    key = _params_key(cfg, info)
    if (cfg["processing"]["reuse_detections"] and not force and csv_p.is_file()
            and meta_p.is_file()):
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("params_key") == key and meta.get("complete"):
            log.info("%s: reutilizando detecções salvas (%s)", info.cam_id, csv_p)
            meta["reused"] = True
            return meta
    if not info.accessible:
        raise RuntimeError(f"{info.cam_id}: vídeo inacessível ({info.path})")

    dcfg, proc = cfg["detector"], cfg["processing"]
    stepper = make_stepper(cfg, info)       # instância nova => detector e ByteTrack próprios desta câmera
    names = stepper.names
    csv_p.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(info.path)
    start = int(proc["start_frame"])
    end = proc["end_frame"] if proc["end_frame"] is not None else info.n_frames
    end = min(int(end), info.n_frames) if info.n_frames else int(end)
    stride = max(1, int(proc["stride"]))
    stats = {"cam": info.cam_id, "frames_read": 0, "frames_processed": 0,
             "frames_without_detections": 0, "read_failures": 0, "rows": 0,
             "untracked_detections": 0, "class_names": names,
             "tracker": cfg["tracker"], "detector": {k: v for k, v in dcfg.items()},
             "params_key": key, "complete": False}
    consecutive = 0
    idx = -1
    with open(csv_p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        while True:
            idx += 1
            if idx >= end:
                break
            if idx < start or (idx - start) % stride:
                if not cap.grab():
                    consecutive += 1
                    stats["read_failures"] += 1
                    if consecutive >= proc["max_read_failures"]:
                        log.error("%s: leitura falhou %d vezes seguidas; interrompendo", info.cam_id, consecutive)
                        break
                else:
                    consecutive = 0
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                consecutive += 1
                stats["read_failures"] += 1
                log.warning("%s: falha de leitura no frame %d", info.cam_id, idx)
                if consecutive >= proc["max_read_failures"]:
                    log.error("%s: leitura falhou %d vezes seguidas; interrompendo", info.cam_id, consecutive)
                    break
                continue
            consecutive = 0
            stats["frames_read"] += 1
            xyxy, conf, cls, ids = stepper.track(frame)
            stats["frames_processed"] += 1
            n = 0
            for i in range(len(xyxy)):
                if ids is None:
                    stats["untracked_detections"] += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                w.writerow([info.cam_id, idx, f"{idx / info.fps:.4f}",
                            f"{info.to_global(idx):.4f}", int(ids[i]), int(cls[i]),
                            names[int(cls[i])], f"{float(conf[i]):.4f}",
                            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
                n += 1
            stats["rows"] += n
            if n == 0:
                stats["frames_without_detections"] += 1
            if stats["frames_processed"] % 200 == 0:
                log.info("%s: %d frames processados", info.cam_id, stats["frames_processed"])
    cap.release()
    stats["complete"] = stats["frames_processed"] > 0
    meta_p.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    return stats


def read_detections(path) -> List[dict]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"cam": r["cam"], "frame": int(r["frame"]), "t_local": float(r["t_local"]),
                         "t_global": float(r["t_global"]), "track_id": int(r["track_id"]),
                         "cls_id": int(r["cls_id"]), "cls_name": r["cls_name"],
                         "conf": float(r["conf"]),
                         "bbox": [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]})
    return rows
