"""Verifica se o ONNX do OSNet-AIN converte para PyTorch com saída idêntica ao onnxruntime e se treina."""
import numpy as np
import onnxruntime as ort
import torch

from reid_training.model import load_backbone

path = "models/osnet_ain_x1_0_vehicle_reid.onnx"
m = load_backbone(path).eval()
print("params", sum(p.numel() for p in m.parameters()),
      "BN2d", sum(1 for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)),
      "BN1d", sum(1 for x in m.modules() if isinstance(x, torch.nn.BatchNorm1d)))
for bs in (1, 5):
    x = np.random.rand(bs, 3, 208, 208).astype("float32")
    ref = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    with torch.no_grad():
        out = m(torch.from_numpy(x)).numpy()
    print("lote", bs, "saida", out.shape, "max|dif| vs onnxruntime:", float(np.abs(out - ref).max()))
m.train()
y = m(torch.from_numpy(np.random.rand(4, 3, 208, 208).astype("float32")))
y.sum().backward()
print("treinavel: grad ok", any(p.grad is not None for p in m.parameters()))
