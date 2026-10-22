"""Exportação de detectores para TensorRT (FP16).

* Ultralytics (YOLOv8/9/10/12...): ONNX pelo exportador oficial -> engine pelo nosso builder (`detectors.build_engine`),
  gravado com o cabeçalho de metadados que o `AutoBackend` do Ultralytics espera (assim `YOLO("x.engine")` funciona).
  (O caminho `format="engine"` do Ultralytics 8.4 exige `nvidia-modelopt` para FP16; evitamos essa dependência.)
* RF-DETR e D-FINE: ONNX com saídas `boxes`/`logits`; engines lidos por `DetrTRTDetector`.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .detectors import build_engine

log = logging.getLogger(__name__)


def export_ultralytics_trt(weights, imgsz, out_dir, fp16=True, workspace_gb=2.0) -> Path:
    """`imgsz`: int ou (alto, largura) múltiplos de 32. Retorna o caminho do .engine."""
    from ultralytics import YOLO
    imgsz = [imgsz, imgsz] if isinstance(imgsz, int) else list(imgsz)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(str(weights)).stem}_{imgsz[0]}x{imgsz[1]}_{'fp16' if fp16 else 'fp32'}"
    engine = out / f"{stem}.engine"
    if engine.is_file():
        return engine
    m = YOLO(str(weights))
    onnx = m.export(format="onnx", imgsz=imgsz, half=False, simplify=True, dynamic=False, batch=1, device=0)
    raw = out / f"{stem}.raw"
    build_engine(onnx, raw, fp16=fp16, workspace_gb=workspace_gb)
    meta = {"description": f"{stem} (mtmc)", "author": "mtmc", "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "version": "mtmc-trt", "stride": int(max(m.model.stride)), "task": "detect", "batch": 1,
            "imgsz": imgsz, "names": {int(k): v for k, v in m.model.names.items()}, "args": {"batch": 1, "half": fp16},
            "channels": 3}
    if getattr(m.model, "end2end", False) or any(getattr(mod, "end2end", False) for mod in m.model.modules()):
        meta["end2end"] = True       # YOLOv10 e similares: NMS embutido
    blob = json.dumps(meta).encode()
    engine.write_bytes(len(blob).to_bytes(4, "little", signed=True) + blob + raw.read_bytes())
    raw.unlink()
    return engine
