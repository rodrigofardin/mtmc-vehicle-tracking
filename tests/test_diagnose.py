import numpy as np
import pytest

from conftest import make_tracklet as T, unit
from mtmc import association as A
from mtmc.association import associate
from mtmc.diagnose import _auc, _best_f1, breakdown, oracle_appearance

E1, E2 = unit(1, 0, 0), unit(0, 1, 0)


def test_oracle_appearance_overrides_and_restores(cfg):
    orig = A.cue_appearance
    a, b, c = T("c001", 1, 0, 10, emb=E1), T("c002", 1, 0, 10, emb=E2), T("c002", 2, 0, 10, emb=E1)
    u2g = {a.uid: 7, b.uid: 7, c.uid: 8}            # a e b são o MESMO veículo no GT, apesar de embeddings opostos
    with oracle_appearance(u2g):
        assert A.cue_appearance(a, b, cfg["association"])["passed"] is True
        assert A.cue_appearance(a, c, cfg["association"])["passed"] is False   # embeddings iguais, mas veículos diferentes
        ids, _ = associate([a, b, c], cfg)
        assert any({a.uid, b.uid} <= set(i.members) for i in ids)
    assert A.cue_appearance is orig


def test_auc_and_best_f1():
    assert _auc([0.9] * 10, [0.1] * 10) == pytest.approx(1.0)
    assert _auc([0.5] * 10, [0.5] * 10) == pytest.approx(0.5)
    assert _auc([1.0], [0.0]) is None                           # amostra pequena demais
    f = _best_f1([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    assert f["f1"] == 1.0 and f["recall"] == 1.0


def test_breakdown_classifies_lost_and_linked_pairs(cfg):
    cfg["association"]["appearance_threshold"] = 0.9
    ts = [
        T("c001", 1, 0, 10, emb=E1), T("c002", 1, 0, 10, emb=E1),                    # mesmo veículo, ligados
        T("c001", 2, 0, 10, emb=E2, cls="car"), T("c002", 2, 0, 10, emb=E2, cls="truck"),  # mesmo veículo, classes diferentes
        T("c001", 3, 0, 10, emb=E1, n=4), T("c002", 3, 0, 10, emb=E1, n=4),          # mesmo veículo, tracklets curtos
        T("c001", 4, 100, 110, emb=E2), T("c002", 4, 0, 10, emb=E2),                  # mesmo veículo, longe no tempo
    ]
    u2g = {"c001:1": 1, "c002:1": 1, "c001:2": 2, "c002:2": 2, "c001:3": 3, "c002:3": 3, "c001:4": 4, "c002:4": 4}
    real = associate(ts, cfg)
    r = breakdown(cfg, ts, u2g, real)["causas_pares_perdidos"]
    assert r["ligado"] == 1
    assert r["classes incompatíveis (ex.: car × truck)"] == 1
    assert r["tracklet curto demais (< min_tracklet_frames)"] == 1
    assert r["intervalo entre tracklets maior que max_gap_s"] == 1
