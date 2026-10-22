import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mtmc.config import DEFAULTS, _merge  # noqa: E402
from mtmc.tracklets import Tracklet  # noqa: E402


def make_video(path, n_frames, fps, size=(64, 48)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for i in range(n_frames):
        img = np.full((size[1], size[0], 3), i % 255, np.uint8)
        vw.write(img)
    vw.release()
    return str(path)


def write(path, text):
    Path(path).write_text(text, encoding="utf-8", newline="")
    return str(path)


@pytest.fixture
def cfg():
    return _merge(DEFAULTS, {})


def make_tracklet(cam, lid, t0, t1, cls="car", emb=None, entry="inside", exit_="inside",
                  n=20, size=(1920, 1080), x0=100.0, dx=10.0):
    ts = np.linspace(t0, t1, n)
    frames = list(range(n))
    boxes = [[x0 + dx * i, 500.0, x0 + dx * i + 100, 600.0] for i in range(n)]
    centers = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes]
    return Tracklet(uid=f"{cam}:{lid}", cam=cam, local_id=lid, cls_name=cls, cls_conf=1.0,
                    first_frame=0, last_frame=n - 1, t_local_first=float(t0), t_local_last=float(t1),
                    t_global_first=float(t0), t_global_last=float(t1), n_obs=n, frames=frames,
                    boxes=boxes, centers=centers, confs=[0.9] * n, mean_conf=0.9,
                    entry=entry, exit=exit_, embedding=None if emb is None else list(map(float, emb)))


def unit(*v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)
