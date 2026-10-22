"""Detectores intercambiáveis (PyTorch/ONNX/TensorRT) com saída unificada.

Todo detector devolve `(xyxy[N,4] float32 em pixels do frame, conf[N], cls[N])` já no espaço de classes
COCO-80 do Ultralytics (car=2, motorcycle=3, bus=5, truck=7), para que o restante do pipeline seja igual.

* `UltralyticsDetector`  — .pt ou .engine (TensorRT) via Ultralytics (YOLOv8/9/10/12...).
* `DetrTRTDetector`      — ONNX/engine de detectores tipo DETR (RF-DETR, D-FINE): saídas `boxes` (cx,cy,w,h
                           normalizados) e `logits` [1, Q, C]; pós-processamento top-k com sigmoide.
* `TRTEngine`/`build_engine` — execução/compilação TensorRT (FP16) sem dependências além de `tensorrt` e `torch`.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

COCO80 = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}   # só as usadas aqui
VEHICLES = ("car", "motorcycle", "bus", "truck")
NAME2COCO80 = {v: k for k, v in COCO80.items()}


class Detector:
    name = "detector"

    def detect(self, frame_bgr: np.ndarray, conf: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


# ----------------------------------------------------------------------------- TensorRT
def build_engine(onnx_path, engine_path, fp16=True, workspace_gb=2.0, log_level="WARNING") -> Path:
    """ONNX -> engine TensorRT (serializado em `engine_path`). Levanta RuntimeError com o log do parser em falha."""
    import tensorrt as trt
    logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("falha ao ler ONNX: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    blob = builder.build_serialized_network(network, cfg)
    if blob is None:
        raise RuntimeError("TensorRT não conseguiu construir o engine (ver log acima)")
    Path(engine_path).write_bytes(bytes(blob))
    log.info("engine %s construído em %.0f s (%.1f MB, fp16=%s)", engine_path, time.time() - t0,
             Path(engine_path).stat().st_size / 1e6, fp16)
    return Path(engine_path)


class TRTEngine:
    """Executa um engine TensorRT usando tensores CUDA do PyTorch como buffers."""

    def __init__(self, path):
        import tensorrt as trt
        import torch
        self.trt, self.torch = trt, torch
        rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = rt.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"engine inválido ou de outra versão do TensorRT/GPU: {path}")
        self.ctx = self.engine.create_execution_context()
        self.in_names, self.out_names = [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            (self.in_names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.out_names).append(n)

    def _torch_dtype(self, name):
        t = self.trt.nptype(self.engine.get_tensor_dtype(name))
        return {np.float32: self.torch.float32, np.float16: self.torch.float16, np.int32: self.torch.int32,
                np.int64: self.torch.int64}[t]

    def __call__(self, x) -> Dict[str, "torch.Tensor"]:
        torch = self.torch
        x = x.contiguous()
        name = self.in_names[0]
        self.ctx.set_input_shape(name, tuple(x.shape))
        self.ctx.set_tensor_address(name, x.data_ptr())
        outs = {}
        for n in self.out_names:
            o = torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=self._torch_dtype(n), device=x.device)
            self.ctx.set_tensor_address(n, o.data_ptr())
            outs[n] = o
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()
        return outs


# ----------------------------------------------------------------------------- Ultralytics
class UltralyticsDetector(Detector):
    def __init__(self, weights, imgsz=640, device="cuda:0", half=True, name=None):
        from ultralytics import YOLO
        self.model = YOLO(str(weights), task="detect")
        self.imgsz, self.device, self.half = imgsz, device, half
        self.name = name or Path(str(weights)).stem
        names = self.model.names
        self.cls_map = {i: NAME2COCO80[n] for i, n in names.items() if n in NAME2COCO80}

    def detect(self, frame_bgr, conf=0.05):
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=conf, iou=0.7, device=self.device,
                               half=self.half, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        xyxy, c, k = b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int)
        keep = np.array([i in self.cls_map for i in k])
        return xyxy[keep].astype(np.float32), c[keep].astype(np.float32), np.array([self.cls_map[i] for i in k[keep]], int)


# ----------------------------------------------------------------------------- DETR (RF-DETR, D-FINE)
class DetrTRTDetector(Detector):
    """Engine TensorRT de um detector DETR-like (boxes cxcywh normalizadas + logits por classe)."""

    def __init__(self, engine, label_names: Dict[int, str], size: Sequence[int], mean, std,
                 boxes_out="boxes", logits_out="logits", topk=300, name="detr", letterbox=False):
        import torch
        self.torch = torch
        self.eng = TRTEngine(engine)
        self.h, self.w = int(size[0]), int(size[1])
        self.mean = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
        self.boxes_out, self.logits_out, self.topk, self.name = boxes_out, logits_out, topk, name
        self.cls_map = {i: NAME2COCO80[n] for i, n in label_names.items() if n in NAME2COCO80}
        self._masks = {}

    def detect(self, frame_bgr, conf=0.05):
        torch = self.torch
        H, W = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.w, self.h), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1)[None].float() / 255.0
        x = (x - self.mean) / self.std
        out = self.eng(x if self.eng.in_names and self.eng._torch_dtype(self.eng.in_names[0]) == torch.float32 else x.half())
        boxes = out[self.boxes_out][0].float()
        logits = out[self.logits_out][0].float()
        scores = logits.sigmoid()
        C = scores.shape[1]
        if C not in self._masks:                                   # classes de veículo no espaço de logits do modelo
            m = torch.zeros(C, device=scores.device)
            for i in self.cls_map:
                if i < C:
                    m[i] = 1.0
            self._masks[C] = m
        top, cls = (scores * self._masks[C]).max(dim=1)            # UM rótulo por consulta (sem duplicatas entre classes)
        keep = top > conf
        top, cls, b = top[keep], cls[keep], boxes[keep]
        cx, cy, bw, bh = b.unbind(-1)
        xyxy = torch.stack([(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H], -1)
        return (xyxy.cpu().numpy().astype(np.float32), top.cpu().numpy().astype(np.float32),
                np.array([self.cls_map[int(i)] for i in cls.cpu().numpy()], int))


# ----------------------------------------------------------------------------- fábrica a partir da configuração YAML
def build_from_config(dcfg: dict, models_dir: Path) -> Detector:
    """`detector.type`: ultralytics (pt/engine via `trt`) | rfdetr | dfine.  Ver configs/*.yaml."""
    kind = dcfg.get("type", "ultralytics")
    models_dir = Path(models_dir)
    spec = {"name": Path(str(dcfg.get("model", dcfg.get("hf", "detector")))).stem, **dcfg}
    if kind == "ultralytics":
        hw = dcfg.get("imgsz", [544, 960])
        hw = [hw, hw] if isinstance(hw, int) else list(hw)
        if dcfg.get("trt", False):
            from .detector_export import export_ultralytics_trt
            eng = export_ultralytics_trt(dcfg["model"], hw, models_dir / "engines")
            return UltralyticsDetector(eng, imgsz=hw, half=False, name=spec["name"] + "-trt")
        return UltralyticsDetector(dcfg["model"], imgsz=max(hw), device=dcfg.get("device", "cuda:0"),
                                   half=bool(dcfg.get("half", True)), name=spec["name"])
    if kind in ("rfdetr", "dfine"):
        from .detector_export_detr import build_detr_detector
        spec["kind"] = kind
        det, _ = build_detr_detector(spec, "trt", models_dir)
        return det
    raise ValueError(f"detector.type desconhecido: {kind!r}")
