"""Avaliação contra ground truth em formato MOTChallenge.

GT esperado: `<gt_dir>/<cam>.txt` com linhas `frame,id,x,y,w,h,...` onde `id` é global
(consistente entre câmeras) e `frame` é 1-based (`frame_offset`=1) como no padrão MOT.
HOTA NÃO é calculado aqui: use `export_mot` + TrackEval (ver README).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_gt(gt_dir, cams, frame_offset=1) -> Dict[str, Dict[int, list]]:
    gt = {}
    for c in cams:
        # layouts aceitos: <gt_dir>/<cam>.txt  ou  <gt_dir>/<cam>/gt/gt.txt (CityFlowV2)
        p = Path(gt_dir) / f"{c}.txt"
        if not p.is_file():
            p = Path(gt_dir) / c / "gt" / "gt.txt"
        if not p.is_file():
            continue
        d = defaultdict(list)
        with open(p, newline="") as f:
            for row in csv.reader(f):
                if len(row) < 6:
                    continue
                fr, i, x, y, w, h = int(float(row[0])) - frame_offset, int(float(row[1])), *map(float, row[2:6])
                d[fr].append((i, [x, y, x + w, y + h]))
        gt[c] = d
    return gt


def iou_matrix(A, B) -> np.ndarray:
    A, B = np.asarray(A, float).reshape(-1, 4), np.asarray(B, float).reshape(-1, 4)
    out = np.zeros((len(A), len(B)))
    for i, a in enumerate(A):
        for j, b in enumerate(B):
            iw = min(a[2], b[2]) - max(a[0], b[0])
            ih = min(a[3], b[3]) - max(a[1], b[1])
            if iw > 0 and ih > 0:
                inter = iw * ih
                out[i, j] = inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return out


def match_frame(gt_items, pr_items, thr):
    """Hungarian por IoU; retorna lista de (gt_id, pred_id, iou)."""
    if not gt_items or not pr_items:
        return []
    M = iou_matrix([b for _, b in gt_items], [b for _, b in pr_items])
    r, c = linear_sum_assignment(-M)
    return [(gt_items[i][0], pr_items[j][0], M[i, j]) for i, j in zip(r, c) if M[i, j] >= thr]


def idf1_and_switches(gt: Dict[str, Dict[int, list]], pred: Dict[str, Dict[int, list]], thr=0.5,
                      key_fn=lambda cam, i: (cam, i)) -> dict:
    """IDF1 + ID switches. `key_fn` define a unicidade da identidade (por câmera ou global)."""
    pair = defaultdict(int)
    n_gt = n_pr = 0
    seq = defaultdict(list)      # gt_key -> [(cam, frame, pred_key)]
    gt_len, pr_len = defaultdict(int), defaultdict(int)
    for cam in gt:
        frames = set(gt[cam]) | set(pred.get(cam, {}))
        for f in sorted(frames):
            g = gt[cam].get(f, [])
            p = pred.get(cam, {}).get(f, [])
            n_gt += len(g)
            n_pr += len(p)
            for i, _ in g:
                gt_len[key_fn(cam, i)] += 1
            for i, _ in p:
                pr_len[key_fn(cam, i)] += 1
            for gi, pi, _ in match_frame(g, p, thr):
                pair[(key_fn(cam, gi), key_fn(cam, pi))] += 1
                seq[key_fn(cam, gi)].append((cam, f, key_fn(cam, pi)))
    gk, pk = sorted(gt_len), sorted(pr_len)
    idtp = 0
    if gk and pk:
        gi_, pi_ = {k: i for i, k in enumerate(gk)}, {k: i for i, k in enumerate(pk)}
        W = np.zeros((len(gk), len(pk)))
        for (g, p), v in pair.items():
            W[gi_[g], pi_[p]] = v
        r, c = linear_sum_assignment(-W)
        idtp = int(W[r, c].sum())
    idfn, idfp = n_gt - idtp, n_pr - idtp
    idsw = 0
    for g, s in seq.items():
        s.sort(key=lambda x: (x[0], x[1]))
        prev = None
        for _, _, p in s:
            if prev is not None and p != prev:
                idsw += 1
            prev = p
    den = 2 * idtp + idfp + idfn
    return {"IDF1": (2 * idtp / den) if den else 0.0, "IDTP": idtp, "IDFP": idfp, "IDFN": idfn,
            "ID_switches": idsw, "gt_detections": n_gt, "pred_detections": n_pr}


def tracklet_to_gt(tracklets, gt, thr=0.5, min_frac=0.5) -> Dict[str, int]:
    """Atribui a cada tracklet o ID global de GT dominante entre suas detecções casadas."""
    out = {}
    for t in tracklets:
        g = gt.get(t.cam)
        if not g:
            continue
        votes = defaultdict(int)
        for f, b in zip(t.frames, t.boxes):
            for gi, _, _ in match_frame(g.get(f, []), [(0, b)], thr):
                votes[gi] += 1
        if votes:
            gi, n = max(votes.items(), key=lambda kv: kv[1])
            if n / len(t.frames) >= min_frac:
                out[t.uid] = gi
    return out


def association_metrics(tracklets, uid2gid: Dict[str, str], uid2gt: Dict[str, int]) -> dict:
    """Pares de tracklets de câmeras DIFERENTES, avaliados contra o GT (só tracklets mapeados)."""
    ts = [t for t in tracklets if t.uid in uid2gt]
    tp = fp = fn = 0
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            a, b = ts[i], ts[j]
            if a.cam == b.cam:
                continue
            same_gt = uid2gt[a.uid] == uid2gt[b.uid]
            same_pr = uid2gid[a.uid] == uid2gid[b.uid]
            tp += same_gt and same_pr
            fp += (not same_gt) and same_pr
            fn += same_gt and (not same_pr)
    return {"pair_precision": tp / (tp + fp) if tp + fp else None,
            "pair_recall": tp / (tp + fn) if tp + fn else None,
            "true_associations": tp, "false_associations": fp, "missed_associations": fn,
            "tracklets_with_gt": len(ts), "tracklets_total": len(tracklets)}


def to_frame_dict(rows: List[dict], id_key) -> Dict[str, Dict[int, list]]:
    d = defaultdict(lambda: defaultdict(list))
    for r in rows:
        d[r["cam"]][r["frame"]].append((id_key(r), r["bbox"]))
    return d


def export_mot(rows: List[dict], id_key, out_dir) -> None:
    """MOTChallenge (frame 1-based) com ID global, utilizável em TrackEval."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    by = defaultdict(list)
    for r in rows:
        by[r["cam"]].append(r)
    for cam, rs in by.items():
        with open(Path(out_dir) / f"{cam}.txt", "w", newline="") as f:
            w = csv.writer(f)
            for r in sorted(rs, key=lambda r: (r["frame"], str(id_key(r)))):
                x1, y1, x2, y2 = r["bbox"]
                w.writerow([r["frame"] + 1, id_key(r), f"{x1:.1f}", f"{y1:.1f}", f"{x2 - x1:.1f}",
                            f"{y2 - y1:.1f}", f"{r.get('conf', 1.0):.3f}", -1, -1, -1])


# ---------------------------------------------------------------------------------------------
# Avaliação global rápida (pré-computa os pareamentos GT x detecções uma única vez)
# ---------------------------------------------------------------------------------------------
def roi_outlier(box, roi) -> bool:
    """Mesma regra do eval.py oficial do AIC22: algum canto da caixa fora da ROI (pixel < 255)."""
    h, w = roi.shape[:2]
    xmin, ymin, xmax, ymax = (int(v) for v in (box[0], box[1], box[2], box[3]))
    for x in (xmin, xmax):
        if 0 <= x < w:
            for y in (ymin, ymax):
                if 0 <= y < h and roi[y, x] < 255:
                    return True
    return False


def load_rois(pattern, cams) -> dict:
    import cv2
    out = {}
    for c in cams:
        p = Path(pattern.format(cam=c))
        if p.is_file():
            im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if im is not None:
                out[c] = im
    return out


class GlobalEvaluator:
    """IDF1 global (entre câmeras) para qualquer mapeamento tracklet -> ID global.

    Protocolo espelhado do AIC22: (1) detecções fora da ROI são descartadas, (2) só IDs globais vistos em
    >= 2 câmeras são avaliados (o GT só anota veículos que passam por >= 2 câmeras). O pareamento GT x
    detecção por frame usa IoU >= limiar (Hungarian). IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN), com as
    identidades casadas por Hungarian sobre as contagens de frames. É uma implementação própria, NÃO o
    eval.py oficial (pytrec_eval/motmetrics não estão instalados); resultados podem diferir levemente.
    """

    def __init__(self, rows, gt, thr=0.5, rois=None):
        self.n_gt = sum(len(v) for fr in gt.values() for v in fr.values())
        self.pair = defaultdict(int)       # (gt_id, uid) -> frames casados
        self.n_det = defaultdict(int)      # uid -> detecções avaliáveis
        self.uid_cam = {}
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            uid = f"{r['cam']}:{r['track_id']}"
            if rois and r["cam"] in rois and roi_outlier(r["bbox"], rois[r["cam"]]):
                continue
            by[r["cam"]][r["frame"]].append((uid, r["bbox"]))
            self.n_det[uid] += 1
            self.uid_cam[uid] = r["cam"]
        for cam, fr in gt.items():
            for f, g in fr.items():
                for gi, uid, _ in match_frame(g, by.get(cam, {}).get(f, []), thr):
                    self.pair[(gi, uid)] += 1

    def idf1(self, uid2gid: Dict[str, str], drop_single_camera=True) -> dict:
        cams = defaultdict(set)
        for u, g in uid2gid.items():
            if u in self.n_det:
                cams[g].add(self.uid_cam[u])
        keep = {g for g, c in cams.items() if len(c) >= 2 or not drop_single_camera}
        n_pred = sum(n for u, n in self.n_det.items() if uid2gid.get(u) in keep)
        cnt = defaultdict(int)
        for (gi, u), v in self.pair.items():
            g = uid2gid.get(u)
            if g in keep:
                cnt[(gi, g)] += v
        idtp = 0
        if cnt:
            gts = sorted({k[0] for k in cnt}); prs = sorted({k[1] for k in cnt})
            gi_, pi_ = {k: i for i, k in enumerate(gts)}, {k: i for i, k in enumerate(prs)}
            W = np.zeros((len(gts), len(prs)))
            for (a, b), v in cnt.items():
                W[gi_[a], pi_[b]] = v
            r, c = linear_sum_assignment(-W)
            idtp = int(W[r, c].sum())
        idfn, idfp = self.n_gt - idtp, n_pred - idtp
        den = 2 * idtp + idfp + idfn
        idp = idtp / (idtp + idfp) if idtp + idfp else 0.0
        idr = idtp / (idtp + idfn) if idtp + idfn else 0.0
        return {"IDF1": 2 * idtp / den if den else 0.0, "IDP": idp, "IDR": idr, "IDTP": idtp,
                "IDFP": idfp, "IDFN": idfn, "pred_ids_kept": len(keep), "gt_detections": self.n_gt,
                "pred_detections_kept": n_pred}
