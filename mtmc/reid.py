"""Interface de Re-ID. Nenhum embedding é fabricado: sem modelo configurado, não há aparência.

Backends reais suportados:
  * onnx        — `onnxruntime` (CPU ou GPU) com um .onnx de entrada NCHW float32.
  * torchscript — arquivo carregável por `torch.jit.load`.
Pré-processamento configurável em `reid.preprocess` (cor, escala, média/desvio). O tamanho de
entrada é lido do próprio modelo ONNX quando estático. Qualquer outro modelo pode ser plugado
implementando `ReIDExtractor.embed`.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import cv2
import numpy as np

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ReIDUnavailable(RuntimeError):
    """Re-ID habilitado, mas modelo/dependência ausentes."""


class Preprocess:
    """BGR uint8 (como o OpenCV lê) -> tensor NCHW float32 conforme a configuração do modelo."""

    def __init__(self, size: Sequence[int], color="rgb", scale=255.0, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.h, self.w = int(size[0]), int(size[1])
        if color not in ("rgb", "bgr"):
            raise ValueError("preprocess.color deve ser rgb|bgr")
        self.color, self.scale = color, float(scale)
        self.mean = np.asarray(mean if mean is not None else (0, 0, 0), np.float32)
        self.std = np.asarray(std if std is not None else (1, 1, 1), np.float32)

    def __call__(self, crops: List[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:
            img = cv2.resize(c, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            if self.color == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x = (img.astype(np.float32) / self.scale - self.mean) / self.std
            out.append(x.transpose(2, 0, 1))
        return np.stack(out).astype(np.float32)


def l2norm(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(len(x), -1)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class ReIDExtractor:
    def embed(self, crops: List[np.ndarray]) -> np.ndarray:  # (N, D), L2-normalizado
        raise NotImplementedError


class OnnxReID(ReIDExtractor):
    def __init__(self, path, preprocess: Optional[dict] = None, size=None, device="cpu", batch=32):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ReIDUnavailable("backend onnx requer `pip install onnxruntime` (ou onnxruntime-gpu)") from e
        avail = ort.get_available_providers()
        prov = [p for p in (["CUDAExecutionProvider"] if str(device).startswith("cuda") else []) if p in avail]
        prov.append("CPUExecutionProvider")
        self.sess = ort.InferenceSession(str(path), providers=prov)
        self.providers = self.sess.get_providers()
        if str(device).startswith("cuda") and "CUDAExecutionProvider" not in self.providers:
            log.warning("Re-ID: CUDA solicitado mas indisponível no onnxruntime; usando CPU")
        inp = self.sess.get_inputs()[0]
        self.inp = inp.name
        shp = inp.shape                               # ex.: [1, 3, 208, 208] ou ['batch', 3, 256, 128]
        self.fixed_batch = shp[0] if isinstance(shp[0], int) else None
        if size is None:
            if not all(isinstance(v, int) for v in shp[2:]):
                raise ReIDUnavailable(f"modelo com entrada dinâmica {shp}: informe reid.input_size [H, W]")
            size = shp[2:]
        self.size, self.batch = tuple(size), batch
        self.pre = Preprocess(size, **(preprocess or {}))

    def embed(self, crops):
        outs = []
        step = self.fixed_batch or self.batch
        for i in range(0, len(crops), step):
            chunk = crops[i:i + step]
            x = self.pre(chunk)
            if self.fixed_batch and len(chunk) < self.fixed_batch:
                x = np.concatenate([x, np.repeat(x[-1:], self.fixed_batch - len(chunk), 0)])
            y = self.sess.run(None, {self.inp: x})[0]
            outs.append(np.asarray(y).reshape(len(x), -1)[:len(chunk)])
        return l2norm(np.concatenate(outs)) if outs else np.zeros((0, 0), np.float32)


class TorchScriptReID(ReIDExtractor):
    def __init__(self, path, preprocess: Optional[dict] = None, size=(256, 128), device="cpu"):
        import torch
        self.torch, self.device = torch, device
        self.model = torch.jit.load(str(path), map_location=device).eval()
        self.pre = Preprocess(size, **(preprocess or {}))

    def embed(self, crops):
        with self.torch.no_grad():
            x = self.torch.from_numpy(self.pre(crops)).to(self.device)
            return l2norm(self.model(x).cpu().numpy())


def build_reid(cfg: dict) -> Optional[ReIDExtractor]:
    """None se Re-ID estiver desabilitado; levanta ReIDUnavailable se habilitado sem modelo."""
    rc = cfg["reid"]
    if not rc.get("enabled"):
        return None
    from pathlib import Path
    if not rc.get("model") or not Path(rc["model"]).is_file():
        raise ReIDUnavailable(f"reid.enabled=true mas reid.model ausente/inexistente: {rc.get('model')!r}")
    size = rc.get("input_size")
    pre = rc.get("preprocess") or {}
    if rc["backend"] == "onnx":
        return OnnxReID(rc["model"], pre, size, rc.get("device", "cpu"))
    if rc["backend"] == "torchscript":
        return TorchScriptReID(rc["model"], pre, size or (256, 128), rc.get("device", "cpu"))
    raise ReIDUnavailable(f"backend de Re-ID desconhecido: {rc['backend']!r}")


def attach_embeddings(tracklets, extractor: ReIDExtractor) -> int:
    """Embedding do tracklet = média L2-normalizada dos embeddings de seus recortes salvos."""
    n = 0
    for t in tracklets:
        imgs = [cv2.imread(p) for p in t.crops]
        imgs = [i for i in imgs if i is not None]
        if not imgs:
            continue
        e = extractor.embed(imgs).mean(axis=0)
        t.embedding = (e / max(np.linalg.norm(e), 1e-12)).tolist()
        n += 1
    return n
