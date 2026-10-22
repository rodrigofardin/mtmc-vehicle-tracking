import math

import numpy as np
import pytest

from mtmc.calibration import image_to_local_meters, parse_aicity_homography
from mtmc.config import load_config
from mtmc.local_tracker import DetArray

pytest.importorskip("ultralytics")


# ---------------------------------------------------------------- DetArray / LocalTracker
def test_detarray_xywh_and_boolean_indexing():
    d = DetArray([[10, 20, 50, 100], [0, 0, 10, 10]], [0.9, 0.3], [2, 7])
    assert d.xywh.tolist() == [[30, 60, 40, 80], [5, 5, 10, 10]]
    sub = d[d.conf > 0.5]
    assert len(sub) == 1 and sub.cls.tolist() == [2.0]


def test_local_tracker_keeps_id_and_trackers_are_independent():
    from mtmc.local_tracker import LocalTracker
    cfg = {"track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.6, "track_buffer": 30,
           "match_thresh": 0.8}
    a, b = LocalTracker(cfg, 10.0), LocalTracker(cfg, 10.0)
    ids_a, ids_b = [], []
    for f in range(6):
        box = [[100 + 5 * f, 100, 180 + 5 * f, 160]]
        for trk, ids in ((a, ids_a), (b, ids_b)):
            out = trk.update(np.array(box), np.array([0.9]), np.array([2]), np.zeros((240, 320, 3), np.uint8))
            ids.extend(o["id"] for o in out)
    assert len(set(ids_a)) == 1 and len(ids_a) >= 3          # mesmo objeto => mesmo ID
    assert ids_a[0] == ids_b[0]                              # ID local igual em trackers diferentes: NÃO são o mesmo objeto
    a.reset()
    out = a.update(np.array([[100, 100, 180, 160]]), np.array([0.9]), np.array([2]), np.zeros((240, 320, 3), np.uint8))
    assert out == [] or out[0]["id"] >= 1


# ---------------------------------------------------------------- config extends / calibração
def test_config_extends_overrides_only_given_fields(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "cameras: [{id: c1, video: v.avi}]\ndetector: {model: m.pt, conf: 0.25, imgsz: 960}\noutput: {dir: out}\n", encoding="utf-8")
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\ndetector: {conf: 0.4}\n", encoding="utf-8")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg["detector"]["conf"] == 0.4 and cfg["detector"]["imgsz"] == 960
    assert cfg["detector"]["model"].endswith("m.pt") and cfg["cameras"][0]["id"] == "c1"


def test_aicity_homography_maps_to_gps_and_meters(tmp_path):
    # homografia sintética: pixel -> (lat, lon) afim; inv(H) é o que a calibração real usa
    lat0, lon0 = 42.5, -90.7
    G = np.array([[0.0, 1e-6, lat0], [1e-6, 0.0, lon0], [0.0, 0.0, 1.0]])    # pixel(x,y) -> (lat, lon)
    H = np.linalg.inv(G)                                                    # arquivo guarda GPS -> pixel
    txt = "Homography matrix: " + ";".join(" ".join(f"{v:.12f}" for v in r) for r in H) + "\nReprojection error: 1.0\n"
    f = tmp_path / "calibration.txt"
    f.write_text(txt, encoding="utf-8")
    M = np.array(image_to_local_meters(parse_aicity_homography(f), (lat0, lon0)))
    p = M @ np.array([1000.0, 0.0, 1.0])                  # x=1000 px => lon + 1e-3 graus
    east = p[0] / p[2]
    assert east == pytest.approx(1e-3 * 111320 * math.cos(math.radians(lat0)), rel=1e-3)


def test_calibration_bad_file_raises(tmp_path):
    f = tmp_path / "c.txt"
    f.write_text("sem homografia aqui\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_aicity_homography(f)
