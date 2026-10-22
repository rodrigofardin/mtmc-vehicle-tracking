"""Avalia um ONNX de Re-ID nos cenários reservados (S01/S02) com recortes do GT.

    python -m reid_training.evaluate --onnx models/osnet_ain_x1_0_vehicle_reid.onnx --eval-data reid_training/data/eval
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from mtmc.reid import OnnxReID
from mtmc.reid_check import retrieval_metrics
from .train import read_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--eval-data", default="reid_training/data/eval")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    root = Path(a.eval_data)
    rows = read_manifest(root)
    ext = OnnxReID(a.onnx, {"color": "rgb", "scale": 255.0, "mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
                   (208, 208), a.device)
    res = {}
    for scene in sorted({r["scene"] for r in rows}):
        rs = [r for r in rows if r["scene"] == scene]
        emb = ext.embed([cv2.imread(str(root / r["path"])) for r in rs])
        res[scene] = retrieval_metrics(emb, [r["cam"] for r in rs], [int(r["label"].split("_")[1]) for r in rs])
        m = res[scene]
        print(f"{scene}: rank1={m['rank1']:.3f} mAP={m['mAP']:.3f} AUC={m['auc']:.3f} pos={m['pos_mean']:.3f} neg={m['neg_mean']:.3f} "
              f"(neg_p90={m['neg_p90']:.3f}, pos_p10={m['pos_p10']:.3f}, {m['queries']} consultas)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
