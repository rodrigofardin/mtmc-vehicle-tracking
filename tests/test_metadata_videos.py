import pytest

from conftest import make_video, write
from mtmc.metadata import MetadataError, read_framenums, read_timestamps
from mtmc.videos import VideoRegistry, probe_video


# ---------------------------------------------------------------- leitura dos .txt
def test_read_timestamps_real_format_crlf(tmp_path):
    p = write(tmp_path / "t.txt", "c001 0\r\nc002 1.640\r\nc003 2.049\r\n")
    t = read_timestamps(p)
    assert t.values == {"c001": 0.0, "c002": 1.64, "c003": 2.049}


def test_read_framenums_integers(tmp_path):
    p = write(tmp_path / "f.txt", "c001 1955\nc002 2110\n")
    assert read_framenums(p).values == {"c001": 1955, "c002": 2110}


def test_tolerant_separators_comments_bom(tmp_path):
    p = tmp_path / "t.txt"
    p.write_bytes("﻿# comentário\nc001\t0\n\nc002, 1.5  # fim\n".encode("utf-8"))
    assert read_timestamps(p).values == {"c001": 0.0, "c002": 1.5}


@pytest.mark.parametrize("text", ["c001 abc\n", "c001\n", "c001 1 2\n", "", "# so comentario\n",
                                  "c001 nan\n"])
def test_invalid_content_raises(tmp_path, text):
    with pytest.raises(MetadataError):
        read_timestamps(write(tmp_path / "t.txt", text))


def test_framenum_rejects_float_and_negative(tmp_path):
    with pytest.raises(MetadataError):
        read_framenums(write(tmp_path / "a.txt", "c001 10.5\n"))
    with pytest.raises(MetadataError):
        read_framenums(write(tmp_path / "b.txt", "c001 -3\n"))


def test_missing_file_raises(tmp_path):
    with pytest.raises(MetadataError):
        read_timestamps(tmp_path / "nao_existe.txt")


def test_duplicate_camera_warns(tmp_path):
    t = read_timestamps(write(tmp_path / "t.txt", "c001 0\nc001 2\n"))
    assert t.values["c001"] == 2.0 and t.warnings


# ---------------------------------------------------------------- registro de vídeos
def _cfg(tmp_path, cams, ts, fn, **extra):
    c = {"cameras": [{"id": i, "video": v} for i, v in cams],
         "metadata": {"timestamp_file": write(tmp_path / "ts.txt", ts),
                      "framenum_file": write(tmp_path / "fn.txt", fn)},
         "timestamp": {"semantics": "start_offset", "confirmed": True},
         "framenum": {"convention": "count"}, "alignment": {"drift_ppm": {}}}
    c.update(extra)
    return c


def test_registry_aligns_different_fps_duration_offsets(tmp_path):
    v1 = make_video(tmp_path / "a.avi", 50, 10)
    v2 = make_video(tmp_path / "b.avi", 60, 20)
    reg = VideoRegistry.from_config(_cfg(tmp_path, [("c001", v1), ("c002", v2)],
                                         "c001 0\nc002 1.5\n", "c001 50\nc002 60\n"))
    a, b = reg["c001"], reg["c002"]
    assert (a.fps, b.fps) == (10, 20)
    assert a.to_global(10) == pytest.approx(1.0)
    assert b.to_global(20) == pytest.approx(1.0 + 1.5)      # t_global = t_local + Δt
    assert b.to_local_frame(2.5) == 20
    # c001 termina em 49/10 = 4.9 s; c002 em 1.5 + 59/20 = 4.45 s
    assert reg.common_timeline() == (pytest.approx(0.0), pytest.approx(4.9))
    assert reg.synchronized_overlap() == (pytest.approx(1.5), pytest.approx(4.45))
    assert not a.warnings and not b.warnings


def test_same_frame_index_is_not_same_instant(tmp_path):
    v = make_video(tmp_path / "a.avi", 20, 10)
    reg = VideoRegistry.from_config(_cfg(tmp_path, [("c001", v), ("c002", v)],
                                         "c001 0\nc002 3\n", "c001 20\nc002 20\n"))
    assert reg["c001"].to_global(5) != reg["c002"].to_global(5)


def test_timestamp_semantics_none_ignores_offsets(tmp_path):
    v = make_video(tmp_path / "a.avi", 20, 10)
    cfg = _cfg(tmp_path, [("c001", v), ("c002", v)], "c001 0\nc002 3\n", "c001 20\nc002 20\n")
    cfg["timestamp"]["semantics"] = "none"
    reg = VideoRegistry.from_config(cfg)
    assert reg["c002"].offset == 0.0


def test_invalid_semantics_rejected(tmp_path):
    v = make_video(tmp_path / "a.avi", 5, 10)
    cfg = _cfg(tmp_path, [("c001", v)], "c001 0\n", "c001 5\n")
    cfg["timestamp"]["semantics"] = "banana"
    with pytest.raises(MetadataError):
        VideoRegistry.from_config(cfg)


def test_framenum_mismatch_detected(tmp_path):
    v = make_video(tmp_path / "a.avi", 30, 10)
    reg = VideoRegistry.from_config(_cfg(tmp_path, [("c001", v)], "c001 0\n", "c001 99\n"),)
    assert any("difere" in w for w in reg["c001"].warnings)


def test_framenum_last_index_convention(tmp_path):
    v = make_video(tmp_path / "a.avi", 30, 10)
    cfg = _cfg(tmp_path, [("c001", v)], "c001 0\n", "c001 29\n")
    cfg["framenum"]["convention"] = "last_index"
    reg = VideoRegistry.from_config(cfg)
    assert reg["c001"].meta_frames == 30 and not reg["c001"].warnings


def test_decoded_frame_count(tmp_path):
    v = make_video(tmp_path / "a.avi", 17, 10)
    info = probe_video("c001", v, count_frames=True)
    assert info.decoded_frames == 17 and info.n_frames == 17


@pytest.mark.parametrize("n", [1, 3, 7])
def test_variable_number_of_cameras(tmp_path, n):
    v = make_video(tmp_path / "a.avi", 10, 10)
    ids = [f"c{i:03d}" for i in range(1, n + 1)]
    ts = "".join(f"{c} {i * 0.5}\n" for i, c in enumerate(ids))
    fn = "".join(f"{c} 10\n" for c in ids)
    reg = VideoRegistry.from_config(_cfg(tmp_path, [(c, v) for c in ids], ts, fn))
    assert reg.cam_ids == ids and len(reg.usable()) == n
    assert reg[ids[-1]].offset == pytest.approx((n - 1) * 0.5)


def test_inaccessible_and_corrupt_video(tmp_path):
    bad = tmp_path / "bad.avi"
    bad.write_bytes(b"isto nao e um video")
    good = make_video(tmp_path / "a.avi", 10, 10)
    reg = VideoRegistry.from_config(_cfg(
        tmp_path, [("c001", good), ("c002", str(tmp_path / "nao.avi")), ("c003", str(bad))],
        "c001 0\nc002 1\nc003 2\n", "c001 10\nc002 10\nc003 10\n"))
    assert reg.usable() == ["c001"]
    assert reg["c002"].warnings and reg["c003"].warnings


def test_missing_and_invalid_metadata_do_not_crash(tmp_path):
    v = make_video(tmp_path / "a.avi", 10, 10)
    cfg = _cfg(tmp_path, [("c001", v), ("c002", v)], "c001 0\n", "c001 xx\n")
    reg = VideoRegistry.from_config(cfg)
    assert any("inválido" in n for n in reg.notes)
    assert any("sem entrada em cam_timestamp" in w for w in reg["c002"].warnings)
    assert reg["c002"].offset == 0.0


def test_camera_only_in_metadata_noted_and_duplicate_rejected(tmp_path):
    v = make_video(tmp_path / "a.avi", 10, 10)
    reg = VideoRegistry.from_config(_cfg(tmp_path, [("c001", v)], "c001 0\nc009 4\n", "c001 10\n"))
    assert any("c009" in n for n in reg.notes)
    with pytest.raises(MetadataError):
        VideoRegistry.from_config(_cfg(tmp_path, [("c001", v), ("c001", v)], "c001 0\n", "c001 10\n"))


def test_empty_camera_list_rejected(tmp_path):
    with pytest.raises(MetadataError):
        VideoRegistry.from_config(_cfg(tmp_path, [], "c001 0\n", "c001 10\n"))


def test_drift_changes_mapping(tmp_path):
    v = make_video(tmp_path / "a.avi", 100, 10)
    cfg = _cfg(tmp_path, [("c001", v)], "c001 0\n", "c001 100\n")
    cfg["alignment"]["drift_ppm"] = {"c001": 1000.0}     # 0,1 %
    reg = VideoRegistry.from_config(cfg)
    assert reg["c001"].to_global(100) == pytest.approx(10.0 * 1.001)
    assert reg["c001"].to_local_frame(reg["c001"].to_global(80)) == 80


def test_uncertain_semantics_is_recorded(tmp_path):
    v = make_video(tmp_path / "a.avi", 10, 10)
    cfg = _cfg(tmp_path, [("c001", v)], "c001 0\n", "c001 10\n")
    cfg["timestamp"]["confirmed"] = False
    assert any("INCERTEZA" in n for n in VideoRegistry.from_config(cfg).notes)
