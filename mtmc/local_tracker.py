"""ByteTrack independente (um por câmera) alimentado por qualquer `Detector` (ver detectors.py).

Usa o `BYTETracker` do Ultralytics diretamente, sem passar pelo `model.track`, para que detectores que não
são Ultralytics (RF-DETR, D-FINE) usem exatamente o mesmo rastreador local dos YOLOs.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List

import numpy as np


class DetArray:
    """Detecções em numpy com a interface mínima que o BYTETracker consome (conf, cls, xywh, indexação booleana)."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = np.asarray(xyxy, np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf, np.float32).reshape(-1)
        self.cls = np.asarray(cls, np.float32).reshape(-1)

    @property
    def xywh(self) -> np.ndarray:
        x1, y1, x2, y2 = self.xyxy.T
        return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=1)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, i):
        return DetArray(self.xyxy[i], self.conf[i], self.cls[i])


class LocalTracker:
    def __init__(self, tracker_cfg: dict, fps: float = 10.0):
        from ultralytics.trackers.byte_tracker import BYTETracker
        args = SimpleNamespace(**{"tracker_type": "bytetrack", "fuse_score": True, **tracker_cfg})
        self._cls, self._args, self._fps = BYTETracker, args, fps
        self.reset()

    def reset(self):
        try:
            self.t = self._cls(self._args, frame_rate=int(round(self._fps)))     # ultralytics <= 8.3
        except TypeError:
            self.t = self._cls(self._args)                                       # ultralytics >= 8.4 (sem frame_rate)

    def update(self, xyxy, conf, cls, frame) -> List[dict]:
        out = self.t.update(DetArray(xyxy, conf, cls), frame)
        res = []
        for r in np.asarray(out).reshape(-1, out.shape[-1] if len(out) else 8):
            res.append({"bbox": [float(v) for v in r[:4]], "id": int(r[4]), "conf": float(r[5]), "cls": int(r[6])})
        return res
