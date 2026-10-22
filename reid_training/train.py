"""Ajuste fino do Re-ID de veículos com o GT do CityFlowV2.

    python -m reid_training.train --data reid_training/data/train --eval-data reid_training/data/eval \\
        --onnx models/osnet_ain_x1_0_vehicle_reid.onnx --out reid_training/runs/ft1

Perdas: entropia cruzada com label smoothing (classificador de IDs após o BNNeck) + triplete batch-hard sobre
embeddings normalizados. Amostragem P×K com positivos de câmeras diferentes. O checkpoint FINAL (última época) é o que
vale: não há seleção de época nos dados de avaliação (S01/S02), apenas acompanhamento.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mtmc.reid_check import retrieval_metrics

from .model import MEAN, SIZE, STD, ReIDNet, embed_paths, load_backbone

log = logging.getLogger("reid_train")


# ----------------------------------------------------------------------------- dados
def read_manifest(root: Path):
    rows = list(csv.DictReader(open(root / "manifest.csv", encoding="utf-8")))
    return rows


class CropDataset(Dataset):
    def __init__(self, root: Path, rows, label2idx):
        self.root, self.rows, self.l2i = root, rows, label2idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = cv2.imread(str(self.root / r["path"]))
        return augment(img), self.l2i[r["label"]], r["cam"]


def augment(img: np.ndarray) -> torch.Tensor:
    h, w = img.shape[:2]
    s = random.uniform(0.85, 1.0)                                   # recorte aleatório (simula caixas do detector)
    ch, cw = int(h * s), int(w * random.uniform(0.85, 1.0))
    y0, x0 = random.randint(0, h - ch), random.randint(0, w - cw)
    img = img[y0:y0 + ch, x0:x0 + cw]
    img = cv2.resize(img, (SIZE[1], SIZE[0]), interpolation=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        img = img[:, ::-1]
    x = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = x * random.uniform(0.75, 1.25) + random.uniform(-0.08, 0.08)       # brilho/contraste
    x = np.clip(x, 0, 1)
    x = (x - MEAN) / STD
    t = torch.from_numpy(x.transpose(2, 0, 1).copy())
    if random.random() < 0.5:                                        # random erasing
        eh, ew = int(SIZE[0] * random.uniform(0.1, 0.3)), int(SIZE[1] * random.uniform(0.1, 0.3))
        y, xx = random.randint(0, SIZE[0] - eh), random.randint(0, SIZE[1] - ew)
        t[:, y:y + eh, xx:xx + ew] = torch.randn(3, eh, ew) * 0.5
    return t


class PKSampler:
    """P veículos × K recortes por lote; prefere 2 câmeras por veículo (positivos entre câmeras)."""

    def __init__(self, rows, p=16, k=4, iters=150, seed=0):
        self.by_label = defaultdict(lambda: defaultdict(list))
        for i, r in enumerate(rows):
            self.by_label[r["label"]][r["cam"]].append(i)
        self.labels = list(self.by_label)
        self.p, self.k, self.iters = p, k, iters
        self.rng = random.Random(seed)

    def __len__(self):
        return self.iters

    def __iter__(self):
        for _ in range(self.iters):
            batch = []
            for lab in self.rng.sample(self.labels, self.p):
                cams = list(self.by_label[lab])
                self.rng.shuffle(cams)
                chosen = cams[:2]
                pool = [i for c in chosen for i in self.by_label[lab][c]]
                if len(pool) >= self.k:
                    # metade de cada câmera quando possível
                    if len(chosen) == 2 and self.k >= 2:
                        half = self.k // 2
                        a = self.rng.sample(self.by_label[lab][chosen[0]], min(half, len(self.by_label[lab][chosen[0]])))
                        b = self.rng.sample(self.by_label[lab][chosen[1]], min(self.k - half, len(self.by_label[lab][chosen[1]])))
                        rest = [i for i in pool if i not in a and i not in b]
                        sel = a + b + self.rng.sample(rest, max(0, min(self.k - len(a) - len(b), len(rest))))
                    else:
                        sel = self.rng.sample(pool, self.k)
                else:
                    sel = [self.rng.choice(pool) for _ in range(self.k)]      # poucos recortes: com reposição
                batch.extend(sel)
            yield batch


def batch_hard_triplet(emb: torch.Tensor, labels: torch.Tensor, margin=0.3) -> torch.Tensor:
    d = torch.cdist(emb, emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    hardest_pos = (d * (same & ~eye)).max(1).values
    hardest_neg = (d + 1e5 * same).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


# ----------------------------------------------------------------------------- avaliação (cenários reservados)
def evaluate(model, eval_root: Path, which: str, device: str) -> dict:
    rows = read_manifest(eval_root)
    out = {}
    for scene in sorted({r["scene"] for r in rows}):
        rs = [r for r in rows if r["scene"] == scene]
        emb = embed_paths(model, [r["path"] for r in rs], which, device, root=eval_root)
        out[scene] = retrieval_metrics(emb, [r["cam"] for r in rs], [int(r["label"].split("_")[1]) for r in rs])
    return out


# ----------------------------------------------------------------------------- treino
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="reid_training/data/train")
    ap.add_argument("--eval-data", default="reid_training/data/eval")
    ap.add_argument("--onnx", default="models/osnet_ain_x1_0_vehicle_reid.onnx")
    ap.add_argument("--out", default="reid_training/runs/ft1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--iters", type=int, default=150, help="iterações por época")
    ap.add_argument("-p", type=int, default=16)
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("--lr-backbone", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(a.data)
    rows = read_manifest(root)
    labels = sorted({r["label"] for r in rows})
    l2i = {l: i for i, l in enumerate(labels)}
    log.info("treino: %d recortes, %d veículos, cenários %s", len(rows), len(labels), sorted({r["scene"] for r in rows}))

    model = ReIDNet(load_backbone(a.onnx), n_classes=len(labels)).to(dev)
    eval_root = Path(a.eval_data)
    history = []
    if eval_root.exists():
        for which in ("feat",):
            base = evaluate(model, eval_root, which, dev)        # linha de base: modelo original (sem ajuste), BN1d novo ainda não usado
        history.append({"epoch": 0, "eval": base})
        log.info("época 0 (modelo original): %s", {s: (round(m["rank1"], 3), round(m["mAP"], 3), round(m["auc"], 3)) for s, m in base.items()})

    ds = CropDataset(root, rows, l2i)
    sampler = PKSampler(rows, a.p, a.k, a.iters, a.seed)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0)
    bb = [p for p in model.backbone.parameters()]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": a.lr_backbone}, {"params": head, "lr": a.lr_head}], weight_decay=5e-4)
    total, warm = a.epochs * a.iters, 3 * a.iters
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    scaler = torch.amp.GradScaler()
    ce = nn.CrossEntropyLoss(label_smoothing=0.1)

    step = 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        agg = defaultdict(float)
        for x, y, _ in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                f, g, logits = model(x)
            l_id = ce(logits.float(), y)
            l_tr = batch_hard_triplet(F.normalize(f.float(), dim=1), y)
            loss = l_id + l_tr
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            agg["loss"] += loss.item(); agg["id"] += l_id.item(); agg["tri"] += l_tr.item()
            agg["acc"] += (logits.argmax(1) == y).float().mean().item()
        n = len(loader)
        rec = {"epoch": ep, **{k: round(v / n, 4) for k, v in agg.items()}, "elapsed_min": round((time.time() - t0) / 60, 1)}
        if eval_root.exists() and (ep % a.eval_every == 0 or ep == a.epochs):
            rec["eval"] = {w: evaluate(model, eval_root, w, dev) for w in ("feat", "bn")}
            for w, res in rec["eval"].items():
                log.info("   eval[%s]: %s", w, {s: (round(m["rank1"], 3), round(m["mAP"], 3), round(m["auc"], 3)) for s, m in res.items()})
        history.append(rec)
        log.info("época %d/%d  loss=%.3f id=%.3f tri=%.3f acc=%.3f  (%.1f min)", ep, a.epochs, rec["loss"], rec["id"], rec["tri"], rec["acc"], rec["elapsed_min"])
        torch.save({"state": model.state_dict(), "labels": labels, "args": vars(a), "epoch": ep}, out / "last.pt")
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    log.info("treino concluído; checkpoint final: %s", out / "last.pt")


if __name__ == "__main__":
    main()
