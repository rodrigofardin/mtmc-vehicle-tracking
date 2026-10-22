"""Associação global entre tracklets de câmeras distintas.

Princípios:
  * Sem calibração/Re-ID/topologia configurada NÃO há evidência decisiva: tracklets ficam
    separados (status `insufficient_evidence`). Classe e proximidade temporal nunca bastam.
  * Pixels de câmeras diferentes nunca são comparados diretamente; o cue espacial só usa
    (a) topologia de zonas configurada ou (b) homografias para plano comum configuradas.
  * Toda decisão (aceita ou não) é registrada com as evidências usadas.
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .tracklets import Tracklet

DECISIVE = ("appearance", "spatial")


@dataclass
class Decision:
    a: str
    b: str
    relation: str                 # simultaneous | transition
    status: str                   # accepted | rejected_* | insufficient_evidence | ambiguous | conflict
    score: Optional[float]
    evidence: Dict = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------- relação temporal
def temporal_relation(a: Tracklet, b: Tracklet, acfg: dict):
    """Retorna (relation, detalhes) ou None se temporalmente incompatíveis."""
    overlap = min(a.t_global_last, b.t_global_last) - max(a.t_global_first, b.t_global_first)
    if overlap >= acfg["min_overlap_s"]:
        return "simultaneous", {"overlap_s": round(overlap, 3)}
    gap = max(a.t_global_first, b.t_global_first) - min(a.t_global_last, b.t_global_last)
    if overlap < 0 and acfg["min_gap_s"] <= gap <= acfg["max_gap_s"]:
        return "transition", {"gap_s": round(gap, 3)}
    return None


def class_compatible(a: Tracklet, b: Tracklet, acfg: dict) -> bool:
    if a.cls_name == b.cls_name:
        return True
    for grp in acfg.get("class_groups", []) or []:
        if a.cls_name in grp and b.cls_name in grp:
            return True
    return False


# ---------------------------------------------------------------- cues
def cue_appearance(a: Tracklet, b: Tracklet, acfg: dict, relation: str = "simultaneous"):
    """Cosseno entre embeddings de Re-ID. `appearance_scale` [lo, hi] mapeia o cosseno para o score [0, 1]
    (a escala do cosseno depende do modelo; ver `reid-check`). Transições podem exigir limiar próprio."""
    if a.embedding is None or b.embedding is None:
        return None
    sim = float(np.dot(a.embedding, b.embedding))
    thr = acfg["appearance_threshold"]
    if relation == "transition" and acfg.get("appearance_threshold_transition") is not None:
        thr = acfg["appearance_threshold_transition"]
    lo, hi = acfg.get("appearance_scale") or (0.0, 1.0)
    score = max(0.0, min(1.0, (sim - lo) / max(hi - lo, 1e-9)))
    return {"score": score, "passed": sim >= thr, "cosine": round(sim, 4), "threshold": thr}


def _project(H, pts):
    p = np.hstack([np.asarray(pts, float), np.ones((len(pts), 1))]) @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def cue_spatial(a: Tracklet, b: Tracklet, relation: str, acfg: dict, spatial: dict):
    """Topologia de zonas e/ou plano de chão comum. Retorna None se nada configurado."""
    results = []
    # (a) topologia de zonas
    rules = [r for r in spatial.get("transitions", []) or []
             if {r["from"].split(":")[0], r["to"].split(":")[0]} == {a.cam, b.cam}
             and r.get("mode", "transition") == relation]
    any_rule_for_pair = any({r["from"].split(":")[0], r["to"].split(":")[0]} == {a.cam, b.cam}
                            for r in spatial.get("transitions", []) or [])
    if any_rule_for_pair:
        if relation == "transition":
            first, second = (a, b) if a.t_global_last <= b.t_global_first else (b, a)
            gap = second.t_global_first - first.t_global_last
            hit = None
            for r in rules:
                if f"{first.cam}:{first.exit}" == r["from"] and f"{second.cam}:{second.entry}" == r["to"] \
                        and r.get("min_s", 0) <= gap <= r.get("max_s", 1e9):
                    hit = r
                    break
            results.append({"kind": "zone_transition", "passed": hit is not None,
                            "score": 1.0 if hit else 0.0, "rule": hit, "gap_s": round(gap, 3),
                            "exit": f"{first.cam}:{first.exit}", "entry": f"{second.cam}:{second.entry}"})
        else:
            # simultâneo: exige regra de sobreposição para o par de câmeras
            results.append({"kind": "zone_overlap", "passed": bool(rules),
                            "score": 1.0 if rules else 0.0, "rule_count": len(rules)})
    # (b) homografias para plano de chão comum
    hs = spatial.get("homographies", {}) or {}
    if a.cam in hs and b.cam in hs and relation == "simultaneous":
        t0 = max(a.t_global_first, b.t_global_first)
        t1 = min(a.t_global_last, b.t_global_last)
        ts = np.linspace(t0, t1, 10)

        pa = _project(hs[a.cam], a.feet_at(ts))
        pb = _project(hs[b.cam], b.feet_at(ts))
        d = float(np.mean(np.linalg.norm(pa - pb, axis=1)))
        dmax = float(spatial.get("max_ground_dist", 3.0))
        results.append({"kind": "ground_plane", "passed": d <= dmax,
                        "score": max(0.0, 1 - d / dmax), "mean_dist": round(d, 3), "max_dist": dmax})
    if not results:
        return None
    thr = acfg["spatial_threshold"]
    passed = all(r["passed"] for r in results) and min(r["score"] for r in results) >= thr
    return {"score": float(np.mean([r["score"] for r in results])), "passed": passed,
            "components": results, "threshold": thr}


# ---------------------------------------------------------------- decisão por par
def evaluate_pair(a: Tracklet, b: Tracklet, cfg: dict) -> Optional[Decision]:
    acfg, spatial = cfg["association"], cfg["spatial"]
    if a.cam == b.cam:
        return None
    if a.n_obs < acfg["min_tracklet_frames"] or b.n_obs < acfg["min_tracklet_frames"]:
        return None
    tr = temporal_relation(a, b, acfg)
    if tr is None:
        return None
    rel, tdet = tr
    ev = {"temporal": {"relation": rel, **tdet}}
    if not class_compatible(a, b, acfg):
        ev["class"] = {"a": a.cls_name, "b": b.cls_name, "passed": False}
        return Decision(a.uid, b.uid, rel, "rejected_class", None, ev, "classes incompatíveis")
    ev["class"] = {"a": a.cls_name, "b": b.cls_name, "passed": True}
    cues = {"appearance": cue_appearance(a, b, acfg, rel), "spatial": cue_spatial(a, b, rel, acfg, spatial)}
    available = {k: v for k, v in cues.items() if v is not None}
    for k in cues:
        ev[k] = cues[k] if cues[k] is not None else {"available": False}
    failed = [k for k, v in available.items() if not v["passed"]]
    if failed:
        return Decision(a.uid, b.uid, rel, "rejected_contradiction", None, ev,
                        f"cue(s) reprovado(s): {failed}")
    if acfg["require_decisive_cue"] and not any(k in available for k in DECISIVE):
        return Decision(a.uid, b.uid, rel, "insufficient_evidence", None, ev,
                        "sem cue decisivo disponível (Re-ID/espacial não configurados)")
    w = acfg["weights"]
    num = sum(w.get(k, 1.0) * v["score"] for k, v in available.items())
    den = sum(w.get(k, 1.0) for k in available)
    score = num / den if den else 0.0
    if score < acfg["accept_score"]:
        return Decision(a.uid, b.uid, rel, "rejected_low_score", score, ev,
                        f"score {score:.3f} < {acfg['accept_score']}")
    return Decision(a.uid, b.uid, rel, "accepted", score, ev, "evidências suficientes")


# ---------------------------------------------------------------- agrupamento
@dataclass
class GlobalIdentity:
    gid: str
    members: List[str]
    links: List[dict]
    uncertain: bool = False
    note: str = ""


def associate(tracklets: List[Tracklet], cfg: dict) -> Tuple[List[GlobalIdentity], List[Decision]]:
    acfg = cfg["association"]
    by_uid = {t.uid: t for t in tracklets}
    decisions: List[Decision] = []
    for a, b in itertools.combinations(tracklets, 2):
        d = evaluate_pair(a, b, cfg)
        if d is not None:
            decisions.append(d)

    # ---- ambiguidade: para cada tracklet e câmera vizinha, os melhores candidatos aceitos
    acc = [d for d in decisions if d.status == "accepted"]
    cand = defaultdict(list)           # (uid, outra_cam) -> [Decision]
    for d in acc:
        cand[(d.a, by_uid[d.b].cam)].append(d)
        cand[(d.b, by_uid[d.a].cam)].append(d)
    ambiguous = set()
    for key, ds in cand.items():
        if len(ds) > 1:
            ds = sorted(ds, key=lambda d: -d.score)
            if ds[0].score - ds[1].score < acfg["ambiguity_margin"]:
                for d in ds:
                    if ds[0].score - d.score < acfg["ambiguity_margin"]:
                        ambiguous.add(id(d))
            else:
                for d in ds[1:]:       # o vencedor claro mantém; os demais viram conflito
                    d.status, d.reason = "conflict", "outro candidato claramente melhor para o mesmo tracklet"
    for d in acc:
        if id(d) in ambiguous:
            d.status, d.reason = "ambiguous", "candidatos de score próximo para o mesmo tracklet"

    # ---- união com restrições (cannot-link por contradição; uma câmera por identidade)
    parent = {t.uid: t.uid for t in tracklets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cannot = {frozenset((d.a, d.b)) for d in decisions
              if d.status in ("rejected_contradiction", "rejected_class")}
    # tracklets simultâneos na mesma câmera são objetos diferentes
    for a, b in itertools.combinations(tracklets, 2):
        if a.cam == b.cam and min(a.t_global_last, b.t_global_last) > max(a.t_global_first, b.t_global_first):
            cannot.add(frozenset((a.uid, b.uid)))
    members = {t.uid: {t.uid} for t in tracklets}
    links = defaultdict(list)
    for d in sorted([d for d in decisions if d.status == "accepted"], key=lambda d: -d.score):
        ra, rb = find(d.a), find(d.b)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        cams_a = {by_uid[u].cam for u in ma}
        cams_b = {by_uid[u].cam for u in mb}
        if cams_a & cams_b:
            d.status, d.reason = "conflict", "fusão geraria duas observações da mesma câmera na identidade"
            continue
        if any(frozenset((x, y)) in cannot for x in ma for y in mb):
            d.status, d.reason = "conflict", "fusão contradiz outra evidência (cannot-link)"
            continue
        parent[rb] = ra
        members[ra] = ma | mb
        links[ra] = links[ra] + links.pop(rb, []) + [asdict(d)]

    amb_uids = {u for d in decisions if d.status in ("ambiguous", "conflict") for u in (d.a, d.b)}
    groups = defaultdict(list)
    for t in tracklets:
        groups[find(t.uid)].append(t.uid)
    identities = []
    for n, (root, uids) in enumerate(sorted(groups.items(), key=lambda kv: min(
            by_uid[u].t_global_first for u in kv[1])), 1):
        unc = any(u in amb_uids for u in uids)
        identities.append(GlobalIdentity(
            gid=f"G{n:04d}", members=sorted(uids), links=links.get(root, []), uncertain=unc,
            note=("associação ambígua/conflitante envolvendo este tracklet" if unc else
                  ("sem associação (evidência insuficiente ou única observação)" if len(uids) == 1 else ""))))
    return identities, decisions


def save_association(identities, decisions, path_dir) -> None:
    d = Path(path_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "global_identities.json").write_text(
        json.dumps([asdict(i) for i in identities], indent=1, default=float), encoding="utf-8")
    with open(d / "association_decisions.jsonl", "w", encoding="utf-8") as f:
        for x in decisions:
            f.write(json.dumps(asdict(x), default=float) + "\n")


def uid_to_gid(identities) -> Dict[str, str]:
    return {u: i.gid for i in identities for u in i.members}
