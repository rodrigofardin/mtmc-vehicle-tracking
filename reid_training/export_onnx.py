"""Exporta o checkpoint treinado para ONNX (entrada `input` NCHW 208x208, saída `output` 512-d) e confere com o PyTorch.

    python -m reid_training.export_onnx --ckpt reid_training/runs/ft1/last.pt --embed bn --out models/osnet_ain_vehicle_reid_ft.onnx
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from .model import SIZE, ReIDNet, load_backbone

log = logging.getLogger("reid_export")


class Wrap(nn.Module):
    def __init__(self, net: ReIDNet, embed: str):
        super().__init__()
        self.net, self.embed = net, embed

    def forward(self, x):
        f, g, _ = self.net(x)
        return f if self.embed == "feat" else g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx-base", default="models/osnet_ain_x1_0_vehicle_reid.onnx")
    ap.add_argument("--embed", choices=["feat", "bn"], default="bn")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    net = ReIDNet(load_backbone(a.onnx_base), n_classes=len(ck["labels"]))
    net.load_state_dict(ck["state"])
    net.eval()
    m = Wrap(net, a.embed).eval()
    x = torch.randn(2, 3, *SIZE)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(m, x, a.out, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}, dynamo=False)
    ref = m(x).detach().numpy()
    got = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    log.info("ONNX salvo em %s; max|dif| vs PyTorch = %.2e (época %s, embed=%s)", a.out, float(np.abs(ref - got).max()), ck["epoch"], a.embed)


if __name__ == "__main__":
    main()
