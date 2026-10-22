"""RF-DETR e D-FINE -> ONNX -> TensorRT FP16 -> `DetrTRTDetector`."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Tuple

from .detectors import DetrTRTDetector, build_engine

log = logging.getLogger(__name__)

# RF-DETR (COCO): o índice do logit é o `category_id` COCO (1..90)
RFDETR_LABELS = {3: "car", 4: "motorcycle", 6: "bus", 8: "truck"}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def _engine_path(models_dir: Path, name: str, size, fp16=True) -> Path:
    d = models_dir / "engines"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{name}_{size[0]}x{size[1]}_{'fp16' if fp16 else 'fp32'}.engine"


def export_rfdetr_onnx(spec: dict, models_dir: Path) -> Tuple[Path, Tuple[int, int]]:
    import rfdetr
    cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}[spec["size"]]
    cwd = os.getcwd()
    os.chdir(models_dir)                                      # o RF-DETR baixa os pesos no diretório atual
    try:
        m = getattr(rfdetr, cls)()
        res = int(m.model_config.resolution)
        shape = tuple(spec.get("shape") or (res, res))
        out = Path(cwd) / models_dir / "onnx" / f"{spec['name']}_{shape[0]}x{shape[1]}"
        onnx = Path(m.export(output_dir=str(out), shape=shape, format="onnx", verbose=False, fp16=False))
    finally:
        os.chdir(cwd)
    return onnx, shape


def export_dfine_onnx(spec: dict, models_dir: Path):
    import torch
    from transformers import AutoImageProcessor, DFineForObjectDetection
    model = DFineForObjectDetection.from_pretrained(spec["hf"]).eval()
    proc = AutoImageProcessor.from_pretrained(spec["hf"])
    h, w = spec.get("size", [640, 640])
    names = {int(k): v for k, v in model.config.id2label.items()}
    mean, std = (proc.image_mean, proc.image_std) if getattr(proc, "do_normalize", False) else ((0, 0, 0), (1, 1, 1))

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            o = self.m(pixel_values=x)
            return o.pred_boxes, o.logits

    onnx = models_dir / "onnx" / f"{spec['name']}_{h}x{w}.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    if not onnx.is_file():
        dummy = torch.zeros(1, 3, h, w)
        torch.onnx.export(Wrap(model), dummy, str(onnx), input_names=["images"], output_names=["boxes", "logits"],
                          opset_version=17, dynamo=False)
    return onnx, (h, w), names, mean, std


def build_detr_detector(spec: dict, backend: str, models_dir: Path):
    if backend != "trt":
        raise ValueError("DETRs são executados apenas via TensorRT")
    models_dir = Path(models_dir)
    rf = spec["kind"] == "rfdetr"
    names = RFDETR_LABELS if rf else None
    mean, std = IMAGENET if rf else ((0, 0, 0), (1, 1, 1))
    bo, lo = ("dets", "labels") if rf else ("boxes", "logits")
    size = tuple(spec["shape"]) if rf and spec.get("shape") else tuple(spec["size"]) if spec.get("size") and not rf else None
    eng = _engine_path(models_dir, spec["name"], size) if size else None
    if eng is None or not eng.is_file():                       # exporta/compila só se o engine ainda não existe
        if rf:
            onnx, size = export_rfdetr_onnx(spec, models_dir)
        else:
            onnx, size, names, mean, std = export_dfine_onnx(spec, models_dir)
        eng = _engine_path(models_dir, spec["name"], size)
        if not eng.is_file():
            build_engine(onnx, eng, fp16=spec.get("fp16", True), workspace_gb=2.0)
    if names is None:                                          # D-FINE: rótulos do config do modelo (não carrega o modelo todo)
        from transformers import AutoConfig
        names = {int(k): v for k, v in AutoConfig.from_pretrained(spec["hf"]).id2label.items()}
        from transformers import AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(spec["hf"])
        mean, std = (proc.image_mean, proc.image_std) if getattr(proc, "do_normalize", False) else ((0, 0, 0), (1, 1, 1))
    det = DetrTRTDetector(eng, names, size, mean, std, boxes_out=bo, logits_out=lo, name=spec["name"])
    return det, str(eng)
