import numpy as np
import pytest

from mtmc.config import DEFAULTS, _merge
from mtmc.live_assoc import LiveAssociator
from mtmc.reid import ReIDExtractor
from mtmc.videos import VideoInfo


class ColorReID(ReIDExtractor):
    """Dublê de teste: embedding = cor média normalizada do recorte (sem relação com o app real)."""

    def embed(self, crops):
        v = np.array([c.reshape(-1, 3).mean(0) for c in crops], np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


class Reg(dict):
    pass


def make(cams=("c001", "c002"), enabled=True):
    cfg = _merge(DEFAULTS, {"association": {"appearance_threshold": 0.9, "min_tracklet_frames": 5},
                            "server": {"live_association": {"enabled": enabled, "interval_s": 60}}})
    reg = Reg({c: VideoInfo(cam_id=c, path="x", accessible=True, fps=10.0, width=320, height=240,
                            reported_frames=1000) for c in cams})
    return LiveAssociator(cfg, reg, ColorReID())


def frame_with(color):
    img = np.zeros((240, 320, 3), np.uint8)
    img[:] = color
    return img


def feed(a, cam, local_id, color, n=12, t0=0.0, loop=0):
    for i in range(n):
        a.update(cam, i, t0 + i / 10, [{"id": local_id, "name": "car", "conf": 0.9, "bbox": [100.0, 80.0, 180.0, 160.0]}],
                 frame_with(color), loop)


def test_same_vehicle_in_two_cameras_gets_one_stable_global_id():
    a = make()
    feed(a, "c001", 1, (200, 30, 30))
    feed(a, "c002", 7, (200, 30, 30))      # mesma aparência, simultâneo
    feed(a, "c002", 8, (10, 200, 10))      # outro veículo
    a._step()
    d = [{"id": 1, "bbox": [0, 0, 1, 1], "gid": None, "uncertain": False}]
    g1 = a.annotate("c001", d)[0]["gid"]
    g2 = a.annotate("c002", [{"id": 7, "gid": None, "uncertain": False}])[0]["gid"]
    other = a.annotate("c002", [{"id": 8, "gid": None, "uncertain": False}])[0]["gid"]
    assert g1 is not None and g1 == g2
    assert other is None                    # sem associação: não ganha ID global visível
    a._step()                               # nova rodada: o ID não muda
    assert a.annotate("c001", [{"id": 1, "gid": None, "uncertain": False}])[0]["gid"] == g1


def test_different_appearance_not_merged_and_reset_on_loop():
    a = make()
    feed(a, "c001", 1, (200, 30, 30))
    feed(a, "c002", 1, (10, 10, 220))
    a._step()
    assert a.annotate("c001", [{"id": 1, "gid": None, "uncertain": False}])[0]["gid"] is None
    feed(a, "c001", 2, (50, 200, 50))
    feed(a, "c002", 2, (50, 200, 50))
    a._step()
    assert a.annotate("c001", [{"id": 2, "gid": None, "uncertain": False}])[0]["gid"] is not None
    a.reset(1)                              # vídeo reiniciou: tudo é descartado
    assert a.annotate("c001", [{"id": 2, "gid": None, "uncertain": False}])[0]["gid"] is None
    assert not a.tracks


def test_short_tracks_and_borders_ignored():
    a = make()
    feed(a, "c001", 1, (200, 30, 30), n=3)
    feed(a, "c002", 1, (200, 30, 30), n=3)
    a._step()
    assert a.stats["active_tracks"] == 0
    a.update("c001", 0, 0.0, [{"id": 5, "name": "car", "conf": 0.9, "bbox": [0.0, 0.0, 90.0, 90.0]}],
             frame_with((1, 2, 3)), 0)       # encostado na borda: sem recorte
    assert not a.tracks["c001:5"].pending


def test_history_is_bounded():
    a = make()
    feed(a, "c001", 1, (9, 9, 9), n=500)
    assert len(a.tracks["c001:1"].frames) <= 400
