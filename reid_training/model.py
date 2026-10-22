"""Modelo de Re-ID: OSNet-AIN (VeRi, via ONNX -> PyTorch) + BNNeck + classificador de IDs (só no treino)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = (208, 208)       # (alto, largura): entrada do vehicle-reid-0001


def load_backbone(onnx_path) -> nn.Module:
    """ONNX (entrada dinâmica) -> nn.Module treinável, com saída conferida contra o onnxruntime (ver probe_convert.py).

    O onnx2torch não infere a forma da entrada do BatchNorm final (após o Gemm) e aborta com "spatial rank == -2";
    nesse caso (forma desconhecida) assumimos um BatchNorm1d, que é o que ele é neste modelo."""
    import onnx
    from onnx2torch import convert
    from onnx2torch.node_converters import batch_norm as bn_mod
    orig = bn_mod.get_shape_from_value_info
    bn_mod.get_shape_from_value_info = lambda vi: (orig(vi) or [0, 0])
    try:
        return convert(onnx.load(str(onnx_path)))
    finally:
        bn_mod.get_shape_from_value_info = orig


class ReIDNet(nn.Module):
    """forward(x) -> (feat, bn_feat, logits|None). Embedding de inferência: `feat` ou `bn_feat` (ver --embed)."""

    def __init__(self, backbone: nn.Module, n_classes: int = 0, dim: int = 512):
        super().__init__()
        self.backbone = backbone
        self.bn = nn.BatchNorm1d(dim)
        self.bn.bias.requires_grad_(False)                      # BNNeck (Luo et al., BoT)
        self.cls = nn.Linear(dim, n_classes, bias=False) if n_classes else None

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        g = self.bn(f)
        return f, g, (self.cls(g) if self.cls is not None else None)


def embed_fn(model: ReIDNet, which: str = "feat"):
    return lambda x: F.normalize(model(x)[0 if which == "feat" else 1], dim=1)


def preprocess_bgr(imgs, size=SIZE) -> torch.Tensor:
    """Lista de recortes BGR uint8 -> tensor NCHW normalizado (igual ao pré-processamento de inferência)."""
    import cv2
    out = []
    for im in imgs:
        x = cv2.cvtColor(cv2.resize(im, (size[1], size[0]), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        out.append(((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))


@torch.no_grad()
def embed_paths(model: ReIDNet, paths, which="feat", device="cuda", batch=128, root: Path | None = None):
    import cv2
    model.eval()
    outs = []
    f = embed_fn(model, which)
    for i in range(0, len(paths), batch):
        imgs = [cv2.imread(str(root / p if root else p)) for p in paths[i:i + batch]]
        outs.append(f(preprocess_bgr(imgs).to(device)).cpu().numpy())
    return np.concatenate(outs)
