"""Diagnóstico da associação global com ground truth: onde o IDF1 se perde?

Experimentos "oráculo" (trocam UMA etapa por uma versão perfeita) e quebra das causas dos pares perdidos.
Tudo usa os tracklets/detecções já gerados por `run` (nenhuma inferência nova).

  A  real                    o pipeline como está
  B  oráculo de associação   cada tracklet recebe o ID do GT  => teto imposto por detector + tracker local
  C  oráculo de aparência    cue de aparência perfeito, regras/espaço/tempo reais  => o que as REGRAS desperdiçam
  C2 oráculo sem espaço      igual a C, sem homografia   => custo/benefício do cue espacial com aparência perfeita
  D  real sem espaço         Re-ID real, sem homografia  => o cue espacial ajuda ou atrapalha com o Re-ID atual

Quebra de causas: para cada par de tracklets de câmeras diferentes que o GT diz ser o MESMO veículo, por que não foi ligado?
"""
from __future__ import annotations

import copy
import json
import logging
import itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import association as A
from .association import associate, temporal_relation, uid_to_gid
from .evaluation import association_metrics
from .tracklets import Tracklet

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------- oráculos
class oracle_appearance:
    """Troca temporariamente o cue de aparência por um perfeito (1 se mesmo veículo do GT; 0 caso contrário)."""

    def __init__(self, uid2gt: Dict[str, int]):
        self.u2g = uid2gt

    def __enter__(self):
        self._orig = A.cue_appearance
        u2g = self.u2g

        def cue(a, b, acfg, relation="simultaneous"):
            same = a.uid in u2g and b.uid in u2g and u2g[a.uid] == u2g[b.uid]
            return {"score": 1.0 if same else 0.0, "passed": bool(same), "cosine": 1.0 if same else 0.0, "threshold": 0.5}

        A.cue_appearance = cue
        return self

    def __exit__(self, *exc):
        A.cue_appearance = self._orig


def _metrics(ge, ts, gmap, u2g, drop_single=True) -> dict:
    g = ge.idf1(gmap, drop_single)
    m = association_metrics(ts, gmap, u2g)
    return {"IDF1": round(g["IDF1"], 4), "IDP": round(g["IDP"], 4), "IDR": round(g["IDR"], 4),
            "pair_precision": None if m["pair_precision"] is None else round(m["pair_precision"], 4),
            "pair_recall": None if m["pair_recall"] is None else round(m["pair_recall"], 4),
            "true_assoc": m["true_associations"], "false_assoc": m["false_associations"], "missed_assoc": m["missed_associations"]}


def run_experiments(cfg: dict, ts: List[Tracklet], ge, u2g: Dict[str, int]) -> dict:
    out = {}
    ids, decs = associate(ts, cfg)
    out["A_real"] = _metrics(ge, ts, uid_to_gid(ids), u2g)
    real = (ids, decs)

    oracle_map = {t.uid: (f"GT{u2g[t.uid]}" if t.uid in u2g else f"U{t.uid}") for t in ts}
    out["B_oraculo_associacao"] = _metrics(ge, ts, oracle_map, u2g)

    with oracle_appearance(u2g):
        i, _ = associate(ts, cfg)
        out["C_oraculo_aparencia"] = _metrics(ge, ts, uid_to_gid(i), u2g)
        nosp = copy.deepcopy(cfg)
        nosp["spatial"]["homographies"] = {}
        i, _ = associate(ts, nosp)
        out["C2_oraculo_aparencia_sem_espaco"] = _metrics(ge, ts, uid_to_gid(i), u2g)
    nosp = copy.deepcopy(cfg)
    nosp["spatial"]["homographies"] = {}
    i, _ = associate(ts, nosp)
    out["D_real_sem_espaco"] = _metrics(ge, ts, uid_to_gid(i), u2g)
    return out, real


# ----------------------------------------------------------------------------- quebra de causas
def _auc(pos: List[float], neg: List[float]) -> Optional[float]:
    if len(pos) < 5 or len(neg) < 5:
        return None
    rs = np.random.default_rng(0)
    a, b = rs.choice(pos, 20000), rs.choice(neg, 20000)
    return float((a > b).mean() + 0.5 * (a == b).mean())


def _best_f1(scores: List[float], labels: List[int], higher_better=True) -> dict:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if not len(s) or y.sum() == 0:
        return {}
    best = {"f1": 0.0}
    for thr in np.unique(np.quantile(s, np.linspace(0.01, 0.99, 120))):
        pred = (s >= thr) if higher_better else (s <= thr)
        tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int(((~pred) & (y == 1)).sum())
        p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        if f1 > best["f1"]:
            best = {"f1": round(f1, 4), "threshold": round(float(thr), 4), "precision": round(p, 4), "recall": round(r, 4)}
    return best


def breakdown(cfg: dict, ts: List[Tracklet], u2g: Dict[str, int], real) -> dict:
    """Causas de perda (pares GT-mesmo não ligados), origem dos falsos links e poder discriminativo de cada cue."""
    acfg = cfg["association"]
    ids, decs = real
    gmap = uid_to_gid(ids)
    dec = {}
    for d in decs:
        dec[(d.a, d.b)] = d
        dec[(d.b, d.a)] = d
    lab = [t for t in ts if t.uid in u2g]
    reasons = Counter()
    gaps_same = []
    false_links = Counter()
    cos_pos, cos_neg, dist_pos, dist_neg = [], [], [], []
    cand_scores = {"cos": [], "dist": [], "y": []}
    rel_count = Counter()
    for a, b in itertools.combinations(lab, 2):
        if a.cam == b.cam:
            continue
        same = u2g[a.uid] == u2g[b.uid]
        linked = gmap[a.uid] == gmap[b.uid]
        d = dec.get((a.uid, b.uid))
        if not same:
            if linked:
                ev = (d.evidence if d else {})
                false_links["falso link (GT diferente, mesmo ID global)"] += 1
                rel_count["falso:" + (ev.get("temporal", {}).get("relation") or "?")] += 1
            if d is not None and d.evidence:
                ap, sp = d.evidence.get("appearance", {}), d.evidence.get("spatial", {})
                if "cosine" in ap:
                    cos_neg.append(ap["cosine"])
                comp = sp.get("components") if isinstance(sp, dict) else None
                if comp:
                    for c in comp:
                        if c.get("kind") == "ground_plane":
                            dist_neg.append(c["mean_dist"])
            continue
        # ---- par positivo (GT: mesmo veículo)
        if linked:                                       # direto ou por cadeia (via um terceiro tracklet)
            reasons["ligado"] += 1
            rel_count["verdadeiro:" + (temporal_relation(a, b, acfg) or ("incompativel",))[0]] += 1
            continue
        if a.n_obs < acfg["min_tracklet_frames"] or b.n_obs < acfg["min_tracklet_frames"]:
            reasons["tracklet curto demais (< min_tracklet_frames)"] += 1
            continue
        tr = temporal_relation(a, b, acfg)
        ov = min(a.t_global_last, b.t_global_last) - max(a.t_global_first, b.t_global_first)
        if tr is None:
            gap = max(a.t_global_first, b.t_global_first) - min(a.t_global_last, b.t_global_last)
            gaps_same.append(gap)
            if ov > 0:
                reasons["sobreposição temporal menor que min_overlap_s"] += 1
            else:
                reasons["intervalo entre tracklets maior que max_gap_s"] += 1
            continue
        rel_count["verdadeiro:" + tr[0]] += 1
        if d is None:
            reasons["sem decisão (outro motivo)"] += 1
            continue
        ev = d.evidence or {}
        ap, sp = ev.get("appearance", {}), ev.get("spatial", {})
        if "cosine" in ap:
            cos_pos.append(ap["cosine"])
        comp = sp.get("components") if isinstance(sp, dict) else None
        spd = None
        if comp:
            for c in comp:
                if c.get("kind") == "ground_plane":
                    spd = c["mean_dist"]
                    dist_pos.append(spd)
        if d.status == "rejected_class":
            reasons["classes incompatíveis (ex.: car × truck)"] += 1
        elif d.status == "rejected_contradiction":
            app_fail = ap.get("passed") is False
            sp_fail = isinstance(sp, dict) and sp.get("passed") is False
            reasons["reprovado: aparência E espaço" if app_fail and sp_fail else "reprovado: só aparência" if app_fail else "reprovado: só espaço" if sp_fail else "reprovado: outro"] += 1
        elif d.status == "rejected_low_score":
            reasons["score combinado abaixo de accept_score"] += 1
        elif d.status in ("ambiguous", "conflict"):
            reasons[f"aceito pelos cues mas bloqueado ({d.status})"] += 1
        elif d.status == "accepted":
            reasons["ligado" if linked else "aceito mas não ligado (outro bloqueio)"] += 1
        else:
            reasons[d.status] += 1
    # poder discriminativo dos cues (pares candidatos com decisão e GT)
    pairs = []
    for a, b in itertools.combinations(lab, 2):
        d = dec.get((a.uid, b.uid))
        if a.cam == b.cam or d is None or d.status == "rejected_class":
            continue
        ap = (d.evidence or {}).get("appearance", {})
        sp = (d.evidence or {}).get("spatial", {})
        comp = sp.get("components") if isinstance(sp, dict) else None
        dist = next((c["mean_dist"] for c in (comp or []) if c.get("kind") == "ground_plane"), None)
        pairs.append((ap.get("cosine"), dist, int(u2g[a.uid] == u2g[b.uid]), d.relation))
    cos = [(c, y) for c, _, y, _ in pairs if c is not None]
    dd = [(x, y) for _, x, y, _ in pairs if x is not None]
    joint = [(c, x, y) for c, x, y, _ in pairs if c is not None and x is not None]
    best_joint = {"f1": 0.0}
    if joint:
        C, X, Y = (np.array(v) for v in zip(*joint))
        for tc in np.linspace(0.05, 0.6, 12):
            for tx in (5, 10, 20, 40, 80, 150, 400):
                pred = (C >= tc) & (X <= tx)
                tp, fp, fn = int((pred & (Y == 1)).sum()), int((pred & (Y == 0)).sum()), int(((~pred) & (Y == 1)).sum())
                p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
                f1 = 2 * p * r / max(p + r, 1e-9)
                if f1 > best_joint["f1"]:
                    best_joint = {"f1": round(f1, 4), "cos_min": round(float(tc), 3), "dist_max_m": tx, "precision": round(p, 4), "recall": round(r, 4)}
    total_pos = sum(v for k, v in reasons.items())
    return {
        "pares_positivos_total": total_pos,
        "causas_pares_perdidos": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "origem_falsos_links": dict(false_links) | {k: v for k, v in rel_count.items() if k.startswith("falso")},
        "relacao_temporal_pares_verdadeiros": {k.split(":")[1]: v for k, v in rel_count.items() if k.startswith("verdadeiro")},
        "nota": "pares positivos = pares de tracklets de câmeras diferentes que o GT diz serem o mesmo veículo; 'ligado' inclui ligações por cadeia",
        "gap_s_pares_verdadeiros_incompativeis": {"n": len(gaps_same), "p50": round(float(np.median(gaps_same)), 1) if gaps_same else None,
                                                  "p90": round(float(np.percentile(gaps_same, 90)), 1) if gaps_same else None},
        "cue_aparencia": {"auc": _auc(cos_pos, cos_neg), "pos_n": len(cos_pos), "neg_n": len(cos_neg),
                          "melhor_limiar_f1": _best_f1([c for c, _ in cos], [y for _, y in cos])},
        "cue_espacial": {"auc_menor_distancia": (None if len(dist_pos) < 5 or len(dist_neg) < 5 else _auc([-x for x in dist_pos], [-x for x in dist_neg])),
                         "dist_m_positivos_p50_p90": [round(float(np.percentile(dist_pos, q)), 1) for q in (50, 90)] if dist_pos else None,
                         "dist_m_negativos_p50_p10": [round(float(np.percentile(dist_neg, q)), 1) for q in (50, 10)] if dist_neg else None,
                         "melhor_limiar_f1": _best_f1([-x for x, _ in dd], [y for _, y in dd])},
        "melhor_combinacao_cos_e_dist_f1": best_joint,
    }


def tracklet_quality(ts: List[Tracklet], u2g: Dict[str, int], ge, gt) -> dict:
    """Fragmentação dos tracklets e cobertura do GT pelas detecções (teto do detector + tracker local)."""
    n_gt_boxes = ge.n_gt
    matched = sum(ge.pair.values())
    frag = defaultdict(set)
    for t in ts:
        if t.uid in u2g:
            frag[(t.cam, u2g[t.uid])].add(t.uid)
    per_track = [len(v) for v in frag.values()]
    # cobertura por (câmera, veículo): quadros do GT casados / quadros do GT
    gt_len = Counter()
    for cam, fr in gt.items():
        for f, items in fr.items():
            for gi, _ in items:
                gt_len[(cam, gi)] += 1
    cov_by = defaultdict(int)
    uid_cam = {t.uid: t.cam for t in ts}
    for (gi, uid), v in ge.pair.items():
        if uid in uid_cam:
            cov_by[(uid_cam[uid], gi)] += v
    cov = [min(cov_by.get(k, 0) / n, 1.0) for k, n in gt_len.items()]
    return {
        "tracklets_total": len(ts), "tracklets_com_gt": len(u2g), "tracklets_sem_gt": len(ts) - len(u2g),
        "gt_boxes": n_gt_boxes, "gt_boxes_casadas_por_alguma_deteccao": int(matched), "cobertura_gt_global": round(matched / max(n_gt_boxes, 1), 4),
        "pares_camera_veiculo_no_gt": len(gt_len),
        "tracklets_por_(camera,veiculo)": {"media": round(float(np.mean(per_track)), 2) if per_track else None, "p90": float(np.percentile(per_track, 90)) if per_track else None,
                                           "fracao_fragmentada(>1)": round(float(np.mean([n > 1 for n in per_track])), 3) if per_track else None},
        "cobertura_por_(camera,veiculo)": {"mediana": round(float(np.median(cov)), 3), "p10": round(float(np.percentile(cov, 10)), 3),
                                           "fracao_com_cobertura<50%": round(float(np.mean([c < 0.5 for c in cov])), 3)},
    }
