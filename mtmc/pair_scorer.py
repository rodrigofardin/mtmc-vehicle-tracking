"""Pontuação APRENDIDA de pares de tracklets (regressão logística) + agrupamento global.

Substitui, de forma opcional (`association.method: learned`), a regra "todos os cues aprovam + média ponderada":
  * Features: aparência (cosseno), classe igual, comprimento/confiança e, com homografia, geometria no plano de chão:
      - simultâneos: distância média/p90 entre as posições, direção e velocidade comparadas, sobreposição;
      - transições: intervalo, distância entre saída e entrada, velocidade implícita, coerência de direção e de velocidade.
  * Dois modelos logísticos (simultâneo × transição), treinados com pares rotulados pelo GT (mesmo veículo?) e prior corrigido.
  * A classe diferente deixa de ser veto: vira uma feature (a mesma caminhonete aparece como car/truck em câmeras distintas).
Nada aqui usa GT na inferência. Treino: `python -m mtmc train-scorer`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from .tracklets import Tracklet

FEATURES = {
    "simultaneous": ["cos", "cos2", "same_cls", "log_len_min", "conf_min", "log_dist_mean", "log_dist_p90", "overlap_s",
                     "dir_cos", "moving", "speed_diff", "has_geo"],
    "transition": ["cos", "cos2", "same_cls", "log_len_min", "conf_min", "log_gap", "log_dist", "log_implied_speed",
                   "dir_cos", "speed_ratio", "speed_e", "has_geo"],
}


# ----------------------------------------------------------------------------- geometria no plano de chão
def _project(H, pts) -> np.ndarray:
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]) @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def ground_track(t: Tracklet, H) -> Tuple[np.ndarray, np.ndarray]:
    """(tempos globais, posições em metros) dos pés do veículo; cache no tracklet (fora dos campos do dataclass)."""
    c = t.__dict__.get("_gp_cache")
    if c is None or c[2] != len(t.frames) or c[3] is not H:
        tg, b = t._arrays()
        foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], axis=1)
        c = (tg, _project(H, foot), len(t.frames), H)
        t.__dict__["_gp_cache"] = c
    return c[0], c[1]


def _vel(tg, P, t0, t1) -> np.ndarray:
    """Velocidade (m/s) por ajuste linear na janela [t0, t1] (mín. 3 pontos; amplia se necessário)."""
    m = (tg >= t0) & (tg <= t1)
    if m.sum() < 3:
        i = int(np.argmin(np.abs(tg - (t0 + t1) / 2)))
        m = np.zeros(len(tg), bool)
        m[max(i - 1, 0):i + 2] = True
    if m.sum() < 2 or np.ptp(tg[m]) < 1e-6:
        return np.zeros(2)
    A = np.vstack([tg[m] - tg[m].mean(), np.ones(m.sum())]).T
    return np.array([np.linalg.lstsq(A, P[m, k], rcond=None)[0][0] for k in (0, 1)])


def _cos(u, v) -> Tuple[float, float, float]:
    su, sv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    return (float(u @ v / (su * sv)) if su > 0.8 and sv > 0.8 else 0.0), su, sv


def pair_features(a: Tracklet, b: Tracklet, rel: str, H: Dict[str, list], overlap: float = 0.0, gap: float = 0.0) -> Dict[str, float]:
    f: Dict[str, float] = {}
    cos = float(np.dot(a.embedding, b.embedding)) if a.embedding is not None and b.embedding is not None else 0.0
    f.update(cos=cos, cos2=cos * cos, same_cls=float(a.cls_name == b.cls_name),
             log_len_min=math.log(max(min(a.n_obs, b.n_obs), 1)), conf_min=min(a.mean_conf, b.mean_conf))
    geo = a.cam in H and b.cam in H
    f["has_geo"] = float(geo)
    if rel == "simultaneous":
        t0, t1 = max(a.t_global_first, b.t_global_first), min(a.t_global_last, b.t_global_last)
        f["overlap_s"] = min(max(overlap, t1 - t0), 20.0)
        d_mean = d_p90 = dir_cos = speed_diff = moving = 0.0
        if geo:
            ta, Pa = ground_track(a, H[a.cam])
            tb, Pb = ground_track(b, H[b.cam])
            ts = np.linspace(t0, t1, 10)
            pa = np.stack([np.interp(ts, ta, Pa[:, k]) for k in (0, 1)], 1)
            pb = np.stack([np.interp(ts, tb, Pb[:, k]) for k in (0, 1)], 1)
            d = np.linalg.norm(pa - pb, axis=1)
            d_mean, d_p90 = float(d.mean()), float(np.percentile(d, 90))
            va, vb = _vel(ta, Pa, t0, t1), _vel(tb, Pb, t0, t1)
            dir_cos, sa, sb = _cos(va, vb)
            moving = float(sa > 0.8 and sb > 0.8)
            speed_diff = abs(sa - sb)
        f.update(log_dist_mean=math.log1p(d_mean) if geo else 0.0, log_dist_p90=math.log1p(d_p90) if geo else 0.0,
                 dir_cos=dir_cos, moving=moving, speed_diff=min(speed_diff, 30.0))
    else:
        e, l = (a, b) if a.t_global_last <= b.t_global_first else (b, a)
        dist = implied = dir_cos = ratio = sp_e = 0.0
        if geo:
            te, Pe = ground_track(e, H[e.cam])
            tl, Pl = ground_track(l, H[l.cam])
            p_end, p_start = Pe[-3:].mean(0), Pl[:3].mean(0)
            dist = float(np.linalg.norm(p_start - p_end))
            implied = dist / max(gap, 0.3)
            ve = _vel(te, Pe, te[-1] - 2.0, te[-1])
            vl = _vel(tl, Pl, tl[0], tl[0] + 2.0)
            disp = p_start - p_end
            dir_cos = _cos(ve, disp)[0] if dist > 2.0 else 0.0
            _, sp_e, sp_l = _cos(ve, vl)
            ratio = min(sp_e, sp_l) / max(sp_e, sp_l, 1e-6)
        f.update(log_gap=math.log1p(max(gap, 0.0)), log_dist=math.log1p(dist), log_implied_speed=math.log1p(min(implied, 80.0)),
                 dir_cos=dir_cos, speed_ratio=ratio, speed_e=min(sp_e, 40.0))
    return f


# ----------------------------------------------------------------------------- regressão logística
@dataclass
class LogReg:
    features: List[str]
    mean: np.ndarray
    std: np.ndarray
    w: np.ndarray
    b: float

    def logit(self, f: Dict[str, float]) -> float:
        x = (np.array([f.get(k, 0.0) for k in self.features]) - self.mean) / self.std
        return float(x @ self.w + self.b)

    def to_dict(self) -> dict:
        return {"features": self.features, "mean": self.mean.tolist(), "std": self.std.tolist(), "w": self.w.tolist(), "b": self.b}

    @classmethod
    def from_dict(cls, d: dict) -> "LogReg":
        return cls(d["features"], np.array(d["mean"]), np.array(d["std"]), np.array(d["w"]), float(d["b"]))


def fit_logreg(rows: List[Dict[str, float]], y: List[int], features: List[str], l2: float = 1.0) -> LogReg:
    X = np.array([[r.get(k, 0.0) for k in features] for r in rows], float)
    y = np.asarray(y, float)
    mean, std = X.mean(0), X.std(0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    pos = max(y.sum(), 1.0)
    neg = max(len(y) - y.sum(), 1.0)
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))              # classes balanceadas no ajuste

    def nll(theta):
        w, b = theta[:-1], theta[-1]
        s = Z @ w + b
        ll = sw * (np.logaddexp(0, s) - y * s)
        g = (sw * (1 / (1 + np.exp(-s)) - y))
        return ll.sum() + 0.5 * l2 * (w @ w), np.concatenate([Z.T @ g + l2 * w, [g.sum()]])

    r = minimize(nll, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B")
    w, b = r.x[:-1], float(r.x[-1])
    b += math.log(pos / neg)                                                    # devolve o prior real (probabilidade calibrada)
    return LogReg(features, mean, std, w, b)


@dataclass
class PairScorer:
    models: Dict[str, LogReg]
    meta: dict

    def logit(self, f: Dict[str, float], rel: str) -> float:
        return self.models[rel].logit(f)

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"meta": self.meta, "models": {k: m.to_dict() for k, m in self.models.items()}}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "PairScorer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: LogReg.from_dict(v) for k, v in d["models"].items()}, d.get("meta", {}))


_CACHE: Dict[str, Tuple[float, PairScorer]] = {}


def load_scorer(path) -> PairScorer:
    p = Path(path)
    m = p.stat().st_mtime
    hit = _CACHE.get(str(p))
    if hit and hit[0] == m:
        return hit[1]
    s = PairScorer.load(p)
    _CACHE[str(p)] = (m, s)
    return s


# ----------------------------------------------------------------------------- treino
def training_pairs(ts: List[Tracklet], u2g: Dict[str, int], cfg: dict) -> List[dict]:
    """Pares de tracklets (câmeras diferentes, ambos com rótulo do GT, temporalmente compatíveis) com a etiqueta 'mesmo veículo'."""
    import itertools
    from .association import temporal_relation
    acfg, H = cfg["association"], cfg["spatial"].get("homographies") or {}
    out = []
    lab = [t for t in ts if t.uid in u2g and t.n_obs >= acfg["min_tracklet_frames"]]
    for a, b in itertools.combinations(lab, 2):
        if a.cam == b.cam:
            continue
        tr = temporal_relation(a, b, acfg)
        if tr is None:
            continue
        rel, tdet = tr
        out.append({"rel": rel, "f": pair_features(a, b, rel, H, tdet.get("overlap_s", 0.0), tdet.get("gap_s", 0.0)),
                    "y": int(u2g[a.uid] == u2g[b.uid]), "ga": u2g[a.uid], "gb": u2g[b.uid]})
    return out


def _auc(scores, y) -> Optional[float]:
    s, y = np.asarray(scores, float), np.asarray(y, int)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) < 3 or len(neg) < 3:
        return None
    rs = np.random.default_rng(0)
    a, b = rs.choice(pos, 20000), rs.choice(neg, 20000)
    return float((a > b).mean() + 0.5 * (a == b).mean())


def fit_scorer(pairs: List[dict], l2: float = 1.0, meta: Optional[dict] = None) -> "PairScorer":
    models = {}
    for rel in FEATURES:
        rows = [p["f"] for p in pairs if p["rel"] == rel]
        y = [p["y"] for p in pairs if p["rel"] == rel]
        if sum(y) < 5 or len(y) - sum(y) < 5:
            raise ValueError(f"poucos exemplos para '{rel}': {sum(y)} positivos, {len(y) - sum(y)} negativos")
        models[rel] = fit_logreg(rows, y, FEATURES[rel], l2)
    return PairScorer(models, meta or {})


def grouped_cv(pairs: List[dict], l2: float = 1.0) -> dict:
    """Validação por veículo (2 dobras): treina só com pares cujos DOIS veículos estão numa dobra e avalia na outra."""
    side = lambda g: (g * 2654435761 >> 7) & 1
    res = {}
    for rel in FEATURES:
        P = [p for p in pairs if p["rel"] == rel]
        auc, f1s = [], []
        for k in (0, 1):
            tr = [p for p in P if side(p["ga"]) == k and side(p["gb"]) == k]
            te = [p for p in P if side(p["ga"]) != k and side(p["gb"]) != k]
            if sum(p["y"] for p in tr) < 5 or sum(p["y"] for p in te) < 3:
                continue
            m = fit_logreg([p["f"] for p in tr], [p["y"] for p in tr], FEATURES[rel], l2)
            z = np.array([m.logit(p["f"]) for p in te])
            y = np.array([p["y"] for p in te])
            auc.append(_auc(z, y))
            pred = z >= 0
            tp, fp, fn = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum()), int(((~pred) & (y == 1)).sum())
            pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
            f1s.append(2 * pr * rc / max(pr + rc, 1e-9))
        res[rel] = {"auc_cv": [round(a, 4) for a in auc if a is not None], "f1_cv@p0.5": [round(f, 4) for f in f1s]}
    return res
