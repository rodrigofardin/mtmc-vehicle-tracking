import json

import numpy as np
import pytest

from conftest import make_tracklet as T, unit
from mtmc.association import associate, save_association, uid_to_gid
from mtmc.evaluation import (association_metrics, export_mot, idf1_and_switches, match_frame,
                             to_frame_dict)
from mtmc.reid import ReIDUnavailable, build_reid

A = unit(1, 0, 0)
B = unit(0, 1, 0)


def status(decs, a, b):
    for d in decs:
        if {d.a, d.b} == {a, b}:
            return d.status
    return None


def test_no_reid_no_spatial_keeps_identities_separate(cfg):
    ts = [T("c001", 1, 0, 10), T("c002", 1, 0, 10)]      # mesma classe, simultâneos
    ids, decs = associate(ts, cfg)
    assert len(ids) == 2
    assert status(decs, "c001:1", "c002:1") == "insufficient_evidence"
    assert "Re-ID" in decs[0].reason


def test_correct_simultaneous_match_with_evidence(cfg):
    ts = [T("c001", 1, 0, 10, emb=A), T("c002", 7, 1, 11, emb=A)]
    ids, decs = associate(ts, cfg)
    assert len(ids) == 1 and set(ids[0].members) == {"c001:1", "c002:7"}
    link = ids[0].links[0]
    assert link["relation"] == "simultaneous"
    assert link["evidence"]["appearance"]["cosine"] == pytest.approx(1.0)
    assert link["evidence"]["temporal"]["overlap_s"] > 0


def test_incorrect_match_rejected_and_objects_stay_distinct(cfg):
    ts = [T("c001", 1, 0, 10, emb=A), T("c002", 1, 0, 10, emb=B)]
    ids, decs = associate(ts, cfg)
    assert len(ids) == 2 and status(decs, "c001:1", "c002:1") == "rejected_contradiction"


def test_two_different_simultaneous_objects_match_correctly(cfg):
    ts = [T("c001", 1, 0, 10, emb=A), T("c001", 2, 0, 10, emb=B),
          T("c002", 1, 0, 10, emb=B), T("c002", 2, 0, 10, emb=A)]
    ids, _ = associate(ts, cfg)
    groups = {frozenset(i.members) for i in ids}
    assert groups == {frozenset({"c001:1", "c002:2"}), frozenset({"c001:2", "c002:1"})}


def test_local_ids_equal_across_cameras_not_assumed_same(cfg):
    ts = [T("c001", 5, 0, 10, emb=A), T("c002", 5, 0, 10, emb=B)]
    ids, _ = associate(ts, cfg)
    assert len(ids) == 2


def test_class_mismatch_rejected(cfg):
    ts = [T("c001", 1, 0, 10, emb=A, cls="car"), T("c002", 1, 0, 10, emb=A, cls="truck")]
    ids, decs = associate(ts, cfg)
    assert len(ids) == 2 and status(decs, "c001:1", "c002:1") == "rejected_class"
    cfg["association"]["class_groups"] = [["car", "truck"]]
    assert len(associate(ts, cfg)[0]) == 1


def test_ambiguous_association_is_rejected_and_flagged(cfg):
    a2 = unit(1, 0.02, 0)
    ts = [T("c001", 1, 0, 10, emb=A), T("c002", 1, 0, 10, emb=A), T("c002", 2, 0, 10, emb=a2)]
    ids, decs = associate(ts, cfg)
    assert len(ids) == 3
    assert {status(decs, "c001:1", "c002:1"), status(decs, "c001:1", "c002:2")} == {"ambiguous"}
    assert all(i.uncertain for i in ids)


def test_clear_best_candidate_wins_over_weaker(cfg):
    weaker = unit(1, 0.8, 0)                      # cos≈0.78 ≥ limiar, mas bem pior que A
    ts = [T("c001", 1, 0, 10, emb=A), T("c002", 1, 0, 10, emb=A), T("c002", 2, 0, 10, emb=weaker)]
    ids, decs = associate(ts, cfg)
    assert {"c001:1", "c002:1"} in [set(i.members) for i in ids]
    assert status(decs, "c001:1", "c002:2") == "conflict"


def test_transition_requires_gap_and_decisive_cue(cfg):
    a, b = T("c001", 1, 0, 10, emb=A), T("c002", 1, 15, 25, emb=A)
    assert len(associate([a, b], cfg)[0]) == 1                     # gap 5 s <= max_gap
    far = T("c002", 1, 100, 110, emb=A)
    ids, decs = associate([a, far], cfg)
    assert len(ids) == 2 and not decs                              # fora de max_gap_s


def test_zone_topology_transition(cfg):
    cfg["spatial"]["transitions"] = [{"from": "c001:east", "to": "c002:west", "min_s": 1, "max_s": 20,
                                      "mode": "transition"}]
    ok = [T("c001", 1, 0, 10, exit_="east"), T("c002", 1, 15, 25, entry="west")]
    ids, decs = associate(ok, cfg)
    assert len(ids) == 1 and ids[0].links[0]["evidence"]["spatial"]["components"][0]["kind"] == "zone_transition"
    bad = [T("c001", 1, 0, 10, exit_="north"), T("c002", 1, 15, 25, entry="west")]
    ids, decs = associate(bad, cfg)
    assert len(ids) == 2 and decs[0].status == "rejected_contradiction"


def test_ground_plane_homography_simultaneous(cfg):
    eye = np.eye(3).tolist()
    cfg["spatial"]["homographies"] = {"c001": eye, "c002": eye}
    cfg["spatial"]["max_ground_dist"] = 50.0
    near = [T("c001", 1, 0, 10, x0=100), T("c002", 1, 0, 10, x0=110)]
    far = [T("c001", 1, 0, 10, x0=100), T("c002", 1, 0, 10, x0=900)]
    assert len(associate(near, cfg)[0]) == 1
    ids, decs = associate(far, cfg)
    assert len(ids) == 2 and decs[0].status == "rejected_contradiction"


def test_same_camera_simultaneous_never_merged(cfg):
    ts = [T("c001", 1, 0, 10, emb=A), T("c001", 2, 0, 10, emb=A), T("c002", 1, 0, 10, emb=A)]
    ids, _ = associate(ts, cfg)
    for i in ids:
        cams = [m.split(":")[0] for m in i.members]
        assert len(cams) == len(set(cams))


@pytest.mark.parametrize("n", [2, 4, 8])
def test_n_cameras_one_object_seen_everywhere(cfg, n):
    ts = [T(f"c{i:03d}", 1, 0.2 * i, 10 + 0.2 * i, emb=A) for i in range(n)]
    ids, _ = associate(ts, cfg)
    assert len(ids) == 1 and len(ids[0].members) == n


def test_short_tracklets_ignored(cfg):
    ts = [T("c001", 1, 0, 10, emb=A, n=3), T("c002", 1, 0, 10, emb=A)]
    assert len(associate(ts, cfg)[0]) == 2


def test_save_association_files(cfg, tmp_path):
    ts = [T("c001", 1, 0, 10, emb=A), T("c002", 1, 0, 10, emb=A)]
    ids, decs = associate(ts, cfg)
    save_association(ids, decs, tmp_path)
    assert json.loads((tmp_path / "global_identities.json").read_text())[0]["links"]
    assert (tmp_path / "association_decisions.jsonl").read_text().strip()
    assert set(uid_to_gid(ids)) == {"c001:1", "c002:1"}


# ---------------------------------------------------------------- Re-ID
def test_reid_disabled_returns_none_and_enabled_without_model_fails(cfg, tmp_path):
    assert build_reid(cfg) is None
    cfg["reid"]["enabled"] = True
    with pytest.raises(ReIDUnavailable):
        build_reid(cfg)
    cfg["reid"]["model"] = str(tmp_path / "nao_existe.onnx")
    with pytest.raises(ReIDUnavailable):
        build_reid(cfg)


# ---------------------------------------------------------------- avaliação
def _pred(rows):   # rows: (cam, frame, id)
    out = []
    for cam, f, i in rows:
        out.append({"cam": cam, "frame": f, "track_id": i, "bbox": [10 * i, 0, 10 * i + 8, 8]})
    return out


def _gt(rows):
    d = {}
    for cam, f, i in rows:
        d.setdefault(cam, {}).setdefault(f, []).append((i, [10 * i, 0, 10 * i + 8, 8]))
    return d


def test_idf1_perfect_and_switch():
    gt = _gt([("c1", f, 1) for f in range(10)])
    perfect = to_frame_dict(_pred([("c1", f, 1) for f in range(10)]), lambda r: r["track_id"])
    r = idf1_and_switches(gt, perfect)
    assert r["IDF1"] == 1.0 and r["ID_switches"] == 0
    # mesmo objeto com id 1 por 5 frames e id 1... (bbox idêntico exige mesma posição): usa id->bbox fixo
    pred = []
    for f in range(10):
        pred.append({"cam": "c1", "frame": f, "track_id": 1 if f < 5 else 9, "bbox": [10, 0, 18, 8]})
    r = idf1_and_switches(gt, to_frame_dict(pred, lambda r: r["track_id"]))
    assert r["ID_switches"] == 1 and r["IDF1"] == pytest.approx(0.5)


def test_global_idf1_penalizes_missing_cross_camera_link():
    gt = _gt([("c1", f, 1) for f in range(10)] + [("c2", f, 1) for f in range(10)])
    pred_ok = _pred([("c1", f, 1) for f in range(10)] + [("c2", f, 1) for f in range(10)])
    key = lambda cam, i: i
    ok = idf1_and_switches(gt, to_frame_dict(pred_ok, lambda r: r["track_id"]), key_fn=key)
    assert ok["IDF1"] == 1.0
    gk = lambda r: (1 if r["cam"] == "c1" else 2)      # mesma boxes, mas ids globais separados
    bad = idf1_and_switches(gt, to_frame_dict(pred_ok, gk), key_fn=key)
    assert bad["IDF1"] < 1.0


def test_association_metrics_counts():
    ts = [T("c001", 1, 0, 1), T("c002", 1, 0, 1), T("c003", 1, 0, 1), T("c001", 2, 0, 1), T("c002", 2, 0, 1)]
    gt = {"c001:1": 10, "c002:1": 10, "c003:1": 10, "c001:2": 20, "c002:2": 20}
    pred = {"c001:1": "G1", "c002:1": "G1", "c003:1": "G9", "c001:2": "G1", "c002:2": "G2"}
    m = association_metrics(ts, pred, gt)
    assert m["true_associations"] == 1          # c001:1–c002:1
    assert m["false_associations"] == 1         # c001:2–c002:1 (c001:2–c001:1 é mesma câmera: ignorado)
    assert m["missed_associations"] == 3        # c003:1–c001:1, c003:1–c002:1, c001:2–c002:2
    assert m["pair_precision"] == pytest.approx(1 / 2)
    assert m["pair_recall"] == pytest.approx(1 / 4)


def test_match_frame_threshold():
    g = [(1, [0, 0, 10, 10])]
    assert match_frame(g, [(5, [0, 0, 10, 10])], 0.5)[0][:2] == (1, 5)
    assert match_frame(g, [(5, [8, 8, 20, 20])], 0.5) == []


def test_export_mot_format(tmp_path):
    export_mot(_pred([("c001", 0, 3)]), lambda r: 7, tmp_path)
    line = (tmp_path / "c001.txt").read_text().strip().split(",")
    assert line[0] == "1" and line[1] == "7"
