import numpy as np
import pytest

from conftest import make_tracklet as T, unit
from mtmc.association import associate
from mtmc.cluster import agglomerate
from mtmc.pair_scorer import FEATURES, LogReg, PairScorer, fit_logreg, pair_features, training_pairs

E1, E2, E3 = unit(1, 0, 0), unit(0, 1, 0), unit(0, 0, 1)


def hand_scorer(path):
    """Modelo manual: só o cosseno importa (peso alto), viés negativo."""
    def m(rel):
        feats = FEATURES[rel]
        w = np.zeros(len(feats))
        w[feats.index("cos")] = 12.0
        return LogReg(feats, np.zeros(len(feats)), np.ones(len(feats)), w, -6.0)
    PairScorer({"simultaneous": m("simultaneous"), "transition": m("transition")}, {"hand": True}).save(path)


# ---------------------------------------------------------------- agrupamento
def test_cluster_links_best_pairs_and_respects_one_per_camera():
    cams = ["a", "b", "a", "b"]
    edges = {(0, 1): 5.0, (2, 3): 4.0, (0, 3): -3.0, (1, 2): -3.0}
    got = sorted(sorted(c) for c in agglomerate(4, cams, edges, tau=0.0))
    assert got == [[0, 1], [2, 3]]
    # nó 1 (cam b) liga a dois nós da MESMA câmera a: só o mais forte entra
    got = sorted(sorted(c) for c in agglomerate(3, ["a", "b", "a"], {(0, 1): 5.0, (1, 2): 4.0}, tau=0.0))
    assert got == [[0, 1], [2]]


def test_cluster_threshold_and_veto():
    assert sorted(map(sorted, agglomerate(2, ["a", "b"], {(0, 1): -1.0}, tau=0.0))) == [[0], [1]]
    assert sorted(map(sorted, agglomerate(2, ["a", "b"], {(0, 1): -1.0}, tau=-2.0))) == [[0, 1]]
    # 0-1 forte, 1-2 forte, mas 0-2 é claramente diferente: o veto impede a cadeia
    cams = ["a", "b", "c"]
    e = {(0, 1): 6.0, (1, 2): 5.0, (0, 2): -9.0}
    assert sorted(map(sorted, agglomerate(3, cams, e, tau=0.0, veto=-4.0))) == [[0, 1], [2]]
    assert sorted(map(sorted, agglomerate(3, cams, e, tau=-5.0, veto=-20.0))) == [[0, 1, 2]]


# ---------------------------------------------------------------- features e regressão
def test_pair_features_simultaneous_and_transition_without_geometry():
    a, b = T("c001", 1, 0, 10, emb=E1), T("c002", 1, 2, 12, emb=E1)
    f = pair_features(a, b, "simultaneous", {}, overlap=8.0)
    assert f["cos"] == pytest.approx(1.0) and f["same_cls"] == 1.0 and f["has_geo"] == 0.0
    assert f["overlap_s"] == pytest.approx(8.0)
    c = T("c002", 2, 15, 25, emb=E2, cls="truck")
    g = pair_features(a, c, "transition", {}, gap=5.0)
    assert g["same_cls"] == 0.0 and g["cos"] == pytest.approx(0.0) and g["log_gap"] == pytest.approx(np.log1p(5.0))


def test_pair_features_ground_plane_distance_and_speed():
    H = np.eye(3).tolist()                                     # pixel = metro
    a = T("c001", 1, 0, 10, emb=E1, x0=100, dx=10)             # anda 10 px por observação
    near = T("c002", 1, 0, 10, emb=E1, x0=105, dx=10)
    far = T("c002", 2, 0, 10, emb=E1, x0=900, dx=-10)          # mesma hora, longe e em sentido oposto
    fn = pair_features(a, near, "simultaneous", {"c001": H, "c002": H}, overlap=10.0)
    ff = pair_features(a, far, "simultaneous", {"c001": H, "c002": H}, overlap=10.0)
    assert fn["has_geo"] == 1.0 and fn["log_dist_mean"] < ff["log_dist_mean"]
    assert fn["dir_cos"] > 0.9 and ff["dir_cos"] < -0.9


def test_logreg_separates_and_keeps_prior():
    rng = np.random.default_rng(0)
    rows, y = [], []
    for _ in range(400):
        pos = rng.random() < 0.1
        rows.append({"cos": float(rng.normal(0.8 if pos else 0.1, 0.1))})
        y.append(int(pos))
    m = fit_logreg(rows, y, ["cos"], l2=1.0)
    z_pos, z_neg = m.logit({"cos": 0.85}), m.logit({"cos": 0.05})
    assert z_pos > 0 > z_neg
    # prior corrigido: no meio da transição a probabilidade reflete que positivos são raros
    assert 1 / (1 + np.exp(-m.logit({"cos": 0.2}))) < 0.5


def test_training_pairs_only_labeled_cross_camera(cfg):
    ts = [T("c001", 1, 0, 10, emb=E1), T("c002", 1, 0, 10, emb=E1), T("c002", 2, 0, 10, emb=E2), T("c001", 9, 0, 10, emb=E3)]
    u2g = {"c001:1": 1, "c002:1": 1, "c002:2": 2}               # c001:9 sem rótulo: não entra
    p = training_pairs(ts, u2g, cfg)
    assert sorted((x["y"] for x in p)) == [0, 1]
    assert all(x["rel"] == "simultaneous" for x in p)


# ---------------------------------------------------------------- associate(method=learned)
def test_associate_learned_groups_similar_and_keeps_traceability(cfg, tmp_path):
    hand_scorer(tmp_path / "s.json")
    cfg["association"].update(method="learned", scorer=str(tmp_path / "s.json"), cluster_threshold=0.0)
    ts = [T("c001", 1, 0, 10, emb=E1), T("c002", 7, 0, 10, emb=E1), T("c001", 2, 0, 10, emb=E2), T("c002", 8, 0, 10, emb=E2)]
    ids, decs = associate(ts, cfg)
    groups = {frozenset(i.members) for i in ids}
    assert groups == {frozenset({"c001:1", "c002:7"}), frozenset({"c001:2", "c002:8"})}
    linked = [i for i in ids if len(i.members) > 1]
    assert all(i.links and i.links[0]["evidence"]["method"] == "learned" for i in linked)
    acc = [d for d in decs if d.status == "accepted"]
    assert len(acc) == 2 and all(d.score > 0.9 for d in acc)
    assert all(d.status in ("accepted", "rejected_learned", "conflict") for d in decs)


def test_associate_learned_never_merges_same_camera_or_dissimilar(cfg, tmp_path):
    hand_scorer(tmp_path / "s.json")
    cfg["association"].update(method="learned", scorer=str(tmp_path / "s.json"))
    ts = [T("c001", 1, 0, 10, emb=E1), T("c001", 2, 0, 10, emb=E1), T("c002", 1, 0, 10, emb=E3)]
    ids, _ = associate(ts, cfg)
    assert len(ids) == 3                                       # iguais na mesma câmera + diferente em outra: nada se funde


def test_default_method_is_rules(cfg):
    assert cfg["association"]["method"] == "rules"
