"""Benchmark de detectores contra o GT do CityFlowV2 (S01 train + S02 validation).

Métricas (veículos, classe-agnóstico, pois o GT não tem classe):
  * recall@IoU0.5 / 0.75 em limiares de confiança 0.25 e 0.5, e por altura do alvo (pequeno/médio/grande);
  * AP50 parcial: o GT só anota veículos vistos em >= 2 câmeras, então a precisão é subestimada para
    TODOS os modelos; use para ranquear, não como AP absoluto;
  * detecções/frame em conf 0.25 (proxy de falsos positivos/ruído para o tracker);
  * latência por frame (pré-processamento + inferência + pós, GPU sincronizada) e VRAM do processo.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluation import iou_matrix, load_gt

log = logging.getLogger(__name__)

SCENES = {
    "S01": ("train/S01", ["c001", "c002", "c003", "c004", "c005"]),
    "S02": ("validation/S02", ["c006", "c007", "c008", "c009"]),
}


def sample_frames(root: Path, scenes: List[str], stride: int):
    """[(cam, video, {frame: [gt_boxes]})] com frames do GT a cada `stride`."""
    out = []
    for s in scenes:
        sub, cams = SCENES[s]
        gt = load_gt(root / sub, cams)
        for cam in cams:
            frames = sorted(gt.get(cam, {}))[::stride]
            out.append((cam, str(root / sub / cam / "vdo.avi"), {f: [b for _, b in gt[cam][f]] for f in frames}))
    return out


def run_detector(det, samples, conf=0.05, warmup=10, lat_conf=0.25, lat_frames=200):
    """Executa `det` nos frames amostrados em `conf` (para as métricas de qualidade). A latência é medida
    à parte, em `lat_conf` (ponto de operação real; com conf baixa o NMS/pós-processamento inflaria o tempo).
    Retorna (lista por frame, latências ms)."""
    import torch
    results, lat = [], []
    n = 0
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    for cam, video, frames in samples:
        wanted = set(frames)
        cap = cv2.VideoCapture(video)
        for f in range(max(wanted) + 1):
            if f not in wanted:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            xyxy, c, k = det.detect(img, conf)
            results.append({"cam": cam, "frame": f, "gt": frames[f], "xyxy": xyxy, "conf": c, "cls": k})
            if len(lat) < lat_frames:                       # medição de latência em conf operacional
                sync()
                t0 = time.perf_counter()
                det.detect(img, lat_conf)
                sync()
                if n >= warmup:
                    lat.append((time.perf_counter() - t0) * 1000)
                n += 1
        cap.release()
    return results, lat


def _match(gt, det_xyxy, thr):
    """Quantos GTs são cobertos (Hungarian por IoU) e quais dets casaram."""
    if len(gt) == 0 or len(det_xyxy) == 0:
        return np.zeros(len(gt), bool), np.zeros(len(det_xyxy), bool)
    M = iou_matrix(gt, det_xyxy)
    r, c = linear_sum_assignment(-M)
    g_hit, d_hit = np.zeros(len(gt), bool), np.zeros(len(det_xyxy), bool)
    for i, j in zip(r, c):
        if M[i, j] >= thr:
            g_hit[i], d_hit[j] = True, True
    return g_hit, d_hit


def metrics(results, lat) -> dict:
    out = {"frames": len(results), "gt_boxes": int(sum(len(r["gt"]) for r in results))}
    for T in (0.25, 0.5):
        for iou in (0.5, 0.75):
            hit = tot = 0
            bins = {"small(<60px)": [0, 0], "medium": [0, 0], "large(>150px)": [0, 0]}
            for r in results:
                m = r["conf"] >= T
                g = np.asarray(r["gt"], float).reshape(-1, 4)
                h, _ = _match(g, r["xyxy"][m], iou)
                hit += h.sum(); tot += len(g)
                if iou == 0.5:
                    hh = g[:, 3] - g[:, 1]
                    for k, sel in (("small(<60px)", hh < 60), ("medium", (hh >= 60) & (hh <= 150)), ("large(>150px)", hh > 150)):
                        bins[k][0] += h[sel].sum(); bins[k][1] += sel.sum()
            out[f"recall@{iou}_conf{T}"] = float(hit / max(tot, 1))
            if iou == 0.5:
                out[f"recall_by_size_conf{T}"] = {k: (float(a / b) if b else None) for k, (a, b) in bins.items()}
        out[f"dets_per_frame_conf{T}"] = float(np.mean([(r["conf"] >= T).sum() for r in results]))
    # AP50 parcial (todas as dets conf >= 0.05, ordenadas por confiança)
    scores, tp = [], []
    for r in results:
        g = np.asarray(r["gt"], float).reshape(-1, 4)
        _, d_hit = _match(g, r["xyxy"], 0.5)
        scores.extend(r["conf"].tolist()); tp.extend(d_hit.tolist())
    order = np.argsort(-np.asarray(scores)); tp = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / max(out["gt_boxes"], 1), ctp / np.maximum(ctp + cfp, 1e-9)
    mrec, mpre = np.concatenate([[0], rec, [1]]), np.concatenate([[1], prec, [0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    out["AP50_partial"] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    if lat:
        out.update(latency_ms_p50=float(np.percentile(lat, 50)), latency_ms_p95=float(np.percentile(lat, 95)),
                   fps_p50=float(1000 / np.percentile(lat, 50)))
    return out


def vram_used_mb() -> float:
    import torch
    free, total = torch.cuda.mem_get_info()
    return (total - free) / 2 ** 20
