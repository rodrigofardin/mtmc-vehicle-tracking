"""Verifica empiricamente um modelo de Re-ID contra o ground truth (sem usar o tracker).

Recorta caixas do GT (IDs globais) de várias câmeras, calcula embeddings e mede, para cada
variante de pré-processamento: separação de similaridade (mesmo veículo em câmeras diferentes x
veículos diferentes), AUC e rank-1/mAP de recuperação entre câmeras. Serve para (a) escolher o
pré-processamento correto do modelo e (b) saber, ANTES da associação, quanto o Re-ID discrimina.
"""
from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .evaluation import load_gt
from .reid import IMAGENET_MEAN, IMAGENET_STD, OnnxReID
from .videos import VideoRegistry

log = logging.getLogger(__name__)

VARIANTS: Dict[str, dict] = {
    "rgb_imagenet": {"color": "rgb", "scale": 255.0, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "bgr_imagenet": {"color": "bgr", "scale": 255.0, "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    "rgb_0_1": {"color": "rgb", "scale": 255.0, "mean": None, "std": None},
    "bgr_0_1": {"color": "bgr", "scale": 255.0, "mean": None, "std": None},
    "rgb_0_255": {"color": "rgb", "scale": 1.0, "mean": None, "std": None},
    "bgr_0_255": {"color": "bgr", "scale": 1.0, "mean": None, "std": None},
}


def sample_gt_crops(cfg, registry: VideoRegistry, per_track=3, min_side=48, seed=0):
    """Retorna [(cam, gid, frame, crop_bgr)] com até `per_track` recortes por (câmera, ID global)."""
    gt = load_gt(cfg["evaluation"]["gt_dir"], registry.cam_ids)
    rng = random.Random(seed)
    out = []
    for cam, frames in gt.items():
        info = registry[cam]
        tracks = defaultdict(list)
        for f, items in frames.items():
            for gid, b in items:
                w, h = b[2] - b[0], b[3] - b[1]
                inside = b[0] > 2 and b[1] > 2 and b[2] < info.width - 2 and b[3] < info.height - 2
                if w >= min_side and h >= min_side and inside:
                    tracks[gid].append((f, b))
        want = {}
        for gid, obs in tracks.items():
            obs.sort()
            idx = np.linspace(0, len(obs) - 1, per_track + 2)[1:-1].astype(int) if len(obs) > per_track else range(len(obs))
            for i in set(int(k) for k in idx):
                want[obs[i][0]] = want.get(obs[i][0], []) + [(gid, obs[i][1])]
        cap = cv2.VideoCapture(info.path)
        for f in range(max(want) + 1 if want else 0):
            if f not in want:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            for gid, b in want[f]:
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                out.append((cam, gid, f, img[max(y1, 0):y2, max(x1, 0):x2].copy()))
        cap.release()
    rng.shuffle(out)
    return out


def retrieval_metrics(emb: np.ndarray, cams: List[str], gids: List[int]) -> dict:
    """Similaridade por cosseno (embeddings já normalizados), consultas entre câmeras diferentes."""
    S = emb @ emb.T
    n = len(gids)
    cams_a, gids_a = np.array(cams), np.array(gids)
    pos, neg = [], []
    rank1, aps = [], []
    for i in range(n):
        other = cams_a != cams_a[i]
        same = (gids_a == gids_a[i]) & other
        diff = gids_a != gids_a[i]
        pos.extend(S[i, same]); neg.extend(S[i, diff & other])
        if not same.any():
            continue
        cand = np.where(other)[0]
        order = cand[np.argsort(-S[i, cand])]
        hits = (gids_a[order] == gids_a[i])
        rank1.append(float(hits[0]))
        cum = np.cumsum(hits)
        aps.append(float((cum[hits] / (np.arange(len(hits))[hits] + 1)).mean()))
    pos, neg = np.array(pos), np.array(neg)
    # AUC = P(sim_pos > sim_neg) por amostragem
    rs = np.random.default_rng(0)
    a, b = rs.choice(pos, 20000), rs.choice(neg, 20000)
    return {"pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()), "pos_p10": float(np.percentile(pos, 10)),
            "neg_p90": float(np.percentile(neg, 90)), "auc": float((a > b).mean() + 0.5 * (a == b).mean()),
            "rank1": float(np.mean(rank1)), "mAP": float(np.mean(aps)), "queries": len(rank1),
            "pos_pairs": int(len(pos)), "neg_pairs": int(len(neg))}


def run_check(cfg: dict, registry: VideoRegistry, per_track=3, variants=None) -> Dict[str, dict]:
    rc = cfg["reid"]
    samples = sample_gt_crops(cfg, registry, per_track)
    cams = [s[0] for s in samples]
    gids = [s[1] for s in samples]
    crops = [s[3] for s in samples]
    log.info("%d recortes de GT (%d veículos, %d câmeras)", len(crops), len(set(gids)), len(set(cams)))
    res = {}
    for name in (variants or VARIANTS):
        ext = OnnxReID(rc["model"], VARIANTS[name], rc.get("input_size"), rc.get("device", "cpu"))
        emb = ext.embed(crops)
        res[name] = retrieval_metrics(emb, cams, gids)
        log.info("%s: %s", name, {k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()})
    return res
