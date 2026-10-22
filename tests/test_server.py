import json
import struct
import time

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from conftest import make_video, write  # noqa: E402
from mtmc.config import DEFAULTS, _merge  # noqa: E402
from mtmc.server import create_app  # noqa: E402


def _cfg(tmp_path, ncams=2, bad=False):
    ids = [f"c{i:03d}" for i in range(1, ncams + 1)]
    cams = []
    out = tmp_path / "out" / "detections"
    out.mkdir(parents=True)
    for i, c in enumerate(ids):
        v = make_video(tmp_path / f"{c}.avi", 40, 10, size=(160, 120))
        cams.append({"id": c, "video": v})
        (out / f"{c}.csv").write_text(
            "cam,frame,t_local,t_global,track_id,cls_id,cls_name,conf,x1,y1,x2,y2\n" +
            "".join(f"{c},{f},{f/10},{f/10},7,2,car,0.9,10,10,50,50\n" for f in range(40)))
    if bad:
        cams.append({"id": "c999", "video": str(tmp_path / "nao.avi")})
        ids.append("c999")
    return _merge(DEFAULTS, {
        "cameras": cams,
        "metadata": {"timestamp_file": write(tmp_path / "t.txt", "".join(f"{c} 0\n" for c in ids)),
                     "framenum_file": write(tmp_path / "f.txt", "".join(f"{c} 40\n" for c in ids))},
        "output": {"dir": str(tmp_path / "out")}, "server": {"mode": "replay", "speed": 1.0}})


def _wait_running(client, n, timeout=10):
    t = time.time()
    while time.time() - t < timeout:
        cams = client.get("/api/cameras").json()
        if sum(c["state"] == "running" for c in cams) >= n:
            return cams
        time.sleep(0.1)
    raise AssertionError(client.get("/api/cameras").json())


@pytest.mark.parametrize("n", [1, 3])
def test_cameras_discovered_dynamically_and_frames_stream(tmp_path, n):
    with TestClient(create_app(_cfg(tmp_path, n))) as c:
        assert c.get("/api/health").json()["cameras"] == n
        cams = _wait_running(c, n)
        assert [x["id"] for x in cams] == [f"c{i:03d}" for i in range(1, n + 1)]
        with c.websocket_connect("/api/cameras/c001/ws") as ws:
            pkt = ws.receive_bytes()
        hl = struct.unpack(">I", pkt[:4])[0]
        h = json.loads(pkt[4:4 + hl])
        assert pkt[4 + hl:6 + hl] == b"\xff\xd8"                       # JPEG
        assert h["cam"] == "c001" and h["width"] == 160 and h["mode"] == "replay"
        if h["detections"]:
            d = h["detections"][0]
            assert d["id"] == 7 and d["name"] == "car" and d["gid"] is None
        assert c.get("/api/cameras/c001/snapshot.jpg").content[:2] == b"\xff\xd8"


def test_unknown_camera_and_inaccessible_video(tmp_path):
    with TestClient(create_app(_cfg(tmp_path, 1, bad=True))) as c:
        assert c.get("/api/cameras/zzz").status_code == 404
        _wait_running(c, 1)
        bad = c.get("/api/cameras/c999").json()
        assert bad["state"] == "unavailable" and bad["error"]
        assert c.get("/api/cameras/c999/snapshot.jpg").status_code == 503
        with pytest.raises(Exception):
            with c.websocket_connect("/api/cameras/zzz/ws"):
                pass
