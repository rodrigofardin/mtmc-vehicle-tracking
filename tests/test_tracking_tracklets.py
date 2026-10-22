import numpy as np
import pytest

from conftest import make_video, make_tracklet, write
from mtmc.config import DEFAULTS, _merge
from mtmc import detect_track
from mtmc.detect_track import read_detections, resolve_classes, run_camera
from mtmc.tracklets import build_tracklets, extract_crops, load_tracklets, save_tracklets
from mtmc.videos import VideoRegistry


class _Arr:
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class FakeYOLO:
    instances = []
    calls = 0

    def __init__(self, path):
        self.names = {0: "person", 2: "car", 7: "truck"}
        self.next_id = 1            # estado de tracker PRÓPRIO desta instância
        self.active = None
        FakeYOLO.instances.append(self)

    def track(self, frame, **kw):
        FakeYOLO.calls += 1
        n = FakeYOLO.calls
        i = len(FakeYOLO.instances)
        del i
        if frame[0, 0, 0] % 5 == 4:     # frames "sem detecção" (marcados no vídeo sintético)
            boxes = type("B", (), {"__len__": lambda s: 0, "id": None})()
        else:
            if self.active is None:
                self.active, self.next_id = self.next_id, self.next_id + 1
            boxes = type("B", (), {"__len__": lambda s: 1, "xyxy": _Arr([[5, 5, 30, 30]]),
                                   "conf": _Arr([0.9]), "cls": _Arr([2]), "id": _Arr([self.active])})()
        return [type("R", (), {"boxes": boxes})()]


@pytest.fixture
def fake_yolo(monkeypatch):
    import ultralytics
    FakeYOLO.instances, FakeYOLO.calls = [], 0
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    return FakeYOLO


def _setup(tmp_path, ncams=3):
    v = make_video(tmp_path / "a.avi", 20, 10)
    ids = [f"c{i:03d}" for i in range(1, ncams + 1)]
    cfg = _merge(DEFAULTS, {
        "cameras": [{"id": c, "video": v} for c in ids],
        "metadata": {"timestamp_file": write(tmp_path / "t.txt", "".join(f"{c} {i}\n" for i, c in enumerate(ids))),
                     "framenum_file": write(tmp_path / "f.txt", "".join(f"{c} 20\n" for c in ids))},
        "detector": {"model": "fake.pt", "classes": ["car"]},
        "output": {"dir": str(tmp_path / "out")}})
    return cfg, VideoRegistry.from_config(cfg)


def test_one_model_and_tracker_state_per_camera(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 3)
    for c in reg.cam_ids:
        run_camera(cfg, reg[c])
    assert len(fake_yolo.instances) == 3                      # nada compartilhado
    # IDs locais iguais em câmeras distintas existem e NÃO foram fundidos pela etapa local
    rows = {c: read_detections(detect_track.det_paths(cfg, c)[0]) for c in reg.cam_ids}
    assert all({r["track_id"] for r in v} == {1} for v in rows.values())
    assert all(r["cam"] == c for c, v in rows.items() for r in v)


def test_export_fields_and_global_time(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 2)
    run_camera(cfg, reg["c002"])
    r = read_detections(detect_track.det_paths(cfg, "c002")[0])[0]
    assert r["cls_name"] == "car" and r["conf"] == 0.9 and r["bbox"] == [5, 5, 30, 30]
    assert r["t_global"] == pytest.approx(r["t_local"] + 1.0)    # offset de c002 = 1


def test_frames_without_detection_counted(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 1)
    s = run_camera(cfg, reg["c001"])
    assert s["frames_without_detections"] > 0 and s["frames_processed"] == 20


def test_reuse_saved_detections_skips_detector(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 1)
    run_camera(cfg, reg["c001"])
    n = len(fake_yolo.instances)
    s = run_camera(cfg, reg["c001"])
    assert s["reused"] and len(fake_yolo.instances) == n
    run_camera(cfg, reg["c001"], force=True)
    assert len(fake_yolo.instances) == n + 1
    cfg["detector"]["conf"] = 0.9                                   # parâmetros mudaram
    assert not run_camera(cfg, reg["c001"]).get("reused")


def test_processing_range_and_stride(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 1)
    cfg["processing"].update(start_frame=4, end_frame=14, stride=2)
    run_camera(cfg, reg["c001"])
    frames = {r["frame"] for r in read_detections(detect_track.det_paths(cfg, "c001")[0])}
    assert frames <= {4, 6, 8, 10, 12} and frames


def test_inaccessible_video_raises(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 1)
    cfg["cameras"][0]["video"] = str(tmp_path / "nada.avi")
    reg = VideoRegistry.from_config(cfg)
    with pytest.raises(RuntimeError):
        run_camera(cfg, reg["c001"])


def test_resolve_classes_from_model_names():
    names = {0: "person", 2: "car", 7: "truck"}
    assert resolve_classes(["car", 7], names) == [2, 7]
    assert resolve_classes(None, names) is None
    with pytest.raises(ValueError):
        resolve_classes(["spaceship"], names)


# ---------------------------------------------------------------- tracklets
def _rows(tid, frames, cls="car", conf=0.9, box=(100, 100, 200, 200)):
    return [{"cam": "c001", "frame": f, "t_local": f / 10, "t_global": f / 10 + 1, "track_id": tid,
             "cls_id": 2, "cls_name": cls, "conf": conf, "bbox": list(box)} for f in frames]


def test_build_tracklets_groups_filters_and_traces(cfg):
    rows = _rows(1, range(10, 30)) + _rows(2, range(0, 3)) + _rows(1, [5], cls="truck", conf=0.3)
    ts = build_tracklets(rows, "c001", (1920, 1080), cfg, "src.csv")
    assert [t.uid for t in ts] == ["c001:1"]                         # id 2 curto demais
    t = ts[0]
    assert (t.first_frame, t.last_frame, t.n_obs) == (5, 29, 21)
    assert t.cls_name == "car" and 0.9 < t.cls_conf < 1
    assert t.t_global_first == pytest.approx(1.5) and t.source == "src.csv"
    assert t.entry == "inside" and t.exit == "inside"


def test_border_and_zone_entry_exit(cfg):
    rows = _rows(1, range(0, 10), box=(0, 100, 50, 200)) + _rows(1, range(10, 20), box=(900, 400, 1000, 500))
    t = build_tracklets(rows, "c001", (1920, 1080), cfg)[0]
    assert t.entry == "border" and t.exit == "inside"
    cfg["spatial"]["zones"] = {"c001": {"S": [[800, 300], [1100, 300], [1100, 600], [800, 600]]}}
    t = build_tracklets(rows, "c001", (1920, 1080), cfg)[0]
    assert t.exit == "S"


def test_tracklet_roundtrip_and_interpolation(tmp_path):
    t = make_tracklet("c001", 1, 0, 10, n=11)
    save_tracklets([t], tmp_path / "t.json")
    t2 = load_tracklets(tmp_path / "t.json")[0]
    assert t2.uid == t.uid and t2.centers == t.centers
    assert t.center_at(5.0)[0] == pytest.approx(t.centers[5][0])
    assert t.center_at(50.0) is None


def test_extract_crops_saves_references(tmp_path, fake_yolo):
    cfg, reg = _setup(tmp_path, 1)
    run_camera(cfg, reg["c001"])
    ts = build_tracklets(read_detections(detect_track.det_paths(cfg, "c001")[0]), "c001", (64, 48), cfg)
    extract_crops(ts, reg, cfg)
    assert ts[0].crops and all(__import__("os").path.isfile(p) for p in ts[0].crops)
