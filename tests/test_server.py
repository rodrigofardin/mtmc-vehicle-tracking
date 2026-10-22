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


# ---------------------------------------------------------------- /api/runtime (somente leitura)
def test_runtime_endpoint_structure_and_no_fake_values(tmp_path):
    with TestClient(create_app(_cfg(tmp_path, 2))) as c:
        _wait_running(c, 2)
        r = c.get("/api/runtime").json()
        assert r["read_only"] is True and r["mode"] == "replay"
        assert r["history"]["supported"] is False                              # backend não guarda histórico
        assert r["models"] == [] or all("role" in m for m in r["models"])      # replay: nenhum modelo executa
        ids = {p["id"] for p in r["parameters"]}
        assert {"detector.conf", "tracker.track_buffer", "association.accept_score", "server.speed"} <= ids
        for p in r["parameters"]:
            assert p["editable"] is False and p["status"] in ("confirmed", "unconfirmed", "diverges")
        # replay não tem tracker nem detector em execução => nada "confirmado" para eles (sem inventar valores)
        tr = next(p for p in r["parameters"] if p["id"] == "tracker.track_buffer")
        assert tr["status"] == "unconfirmed" and tr["applied"] is None
        assert [x["id"] for x in r["cameras"]] == ["c001", "c002"]
        assert "video.fps" in r["cameras"][0]["params"]
        # consultar não interrompe o processamento
        a = c.get("/api/cameras").json()[0]["frames_published"]
        for _ in range(5):
            c.get("/api/runtime")
        time.sleep(0.5)
        assert c.get("/api/cameras").json()[0]["frames_published"] >= a


def test_runtime_detects_config_changed_on_disk(tmp_path):
    from mtmc.config import load_config
    base = _cfg(tmp_path, 1)
    import yaml
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump({"cameras": base["cameras"], "metadata": base["metadata"],
                                    "output": {"dir": str(tmp_path / "out")}, "server": {"mode": "replay"},
                                    "association": {"accept_score": 0.5}}), encoding="utf-8")
    cfg = load_config(path)
    with TestClient(create_app(cfg)) as c:
        r = c.get("/api/runtime").json()
        assert r["config"]["changed_on_disk"] is False and r["config"]["file"] == str(path.resolve())
        d = yaml.safe_load(path.read_text(encoding="utf-8"))
        d["association"]["accept_score"] = 0.8
        path.write_text(yaml.safe_dump(d), encoding="utf-8")
        r2 = c.get("/api/runtime").json()
        assert r2["config"]["changed_on_disk"] is True
        assert any(x["key"] == "association.accept_score" and x["running"] == 0.5 and x["on_disk"] == 0.8 for x in r2["config"]["diff"])
        p = next(p for p in r2["parameters"] if p["id"] == "association.accept_score")
        assert p["configured"] == 0.8                       # valor do arquivo; "aplicado" segue não confirmado (replay)
