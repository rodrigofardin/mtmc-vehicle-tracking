"""Carregamento e normalização da configuração YAML."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS = {
    "cameras": [],
    "metadata": {"timestamp_file": None, "framenum_file": None, "location_image": None},
    "timestamp": {"semantics": "start_offset", "confirmed": False},
    "framenum": {"convention": "count"},
    "alignment": {"drift_ppm": {}},
    "detector": {"model": "models/yolov8n.pt", "device": "cpu", "imgsz": 640,
                 "conf": 0.25, "iou": 0.7, "classes": None, "half": False,
                 # Em CPU, neste ambiente (torch 2.4.0+cpu / Windows) o backend oneDNN produziu
                 # saídas não determinísticas e absurdas; desativar o corrige (ver README).
                 "disable_mkldnn": True},
    "tracker": {"track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.6,
                "track_buffer": 30, "match_thresh": 0.8, "fuse_score": True},
    "processing": {"start_frame": 0, "end_frame": None, "stride": 1,
                   "max_read_failures": 30, "reuse_detections": True, "count_frames": False},
    "tracklets": {"min_length": 5, "border_margin": 0.03, "crops_per_tracklet": 3,
                  "crop_dir": "crops"},
    "reid": {"enabled": False, "model": None, "backend": "onnx", "input_size": None,  # None = lê do modelo
             "device": "cpu",
             "preprocess": {"color": "rgb", "scale": 255.0, "mean": [0.485, 0.456, 0.406],
                            "std": [0.229, 0.224, 0.225]}},
    "spatial": {"zones": {}, "transitions": [], "homographies": {}, "layout": {},
                # {pattern: ".../{cam}/calibration.txt", ref_latlon: [lat, lon]} => homografias em metros
                "calibration": None, "max_ground_dist": 3.0},
    "association": {
        "min_overlap_s": 0.5, "max_gap_s": 30.0, "min_gap_s": 0.0,
        "class_compatibility": "strict",
        "weights": {"appearance": 1.0, "spatial": 1.0},
        "appearance_threshold": 0.7, "appearance_threshold_transition": None,
        "appearance_scale": [0.0, 1.0], "spatial_threshold": 0.5,
        "accept_score": 0.7, "ambiguity_margin": 0.1,
        "require_decisive_cue": True, "min_tracklet_frames": 10,
    },
    "output": {"dir": "outputs"},
    # Serviço de streaming para o frontend (modo `serve`)
    "server": {"mode": "live",            # live = YOLO+ByteTrack em tempo real | replay = resultados salvos
               "host": "0.0.0.0", "port": 8000, "stream_width": 960, "jpeg_quality": 70,
               "speed": 1.0,              # velocidade do relógio global (1.0 = tempo real)
               "environment": "local",
               # associação global incremental ao vivo (usa reid + spatial da configuração)
               "live_association": {"enabled": False, "interval_s": 2.0, "retention_s": 60.0, "max_crops": 6,
                                    "min_crop_side": 40}},
    "visualization": {"cell_width": 640, "fps": None, "mode": "global", "draw_trails": True,
                      "max_seconds": None, "cameras": None},
    "evaluation": {"gt_dir": None, "iou_threshold": 0.5,
                   # {cam} => roi.jpg do dataset (região onde os veículos anotados aparecem)
                   "roi_pattern": None, "drop_single_camera": True},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and out[k]:
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw(p: Path) -> dict:
    """Lê o YAML; `extends: outro.yaml` (na mesma pasta) herda e sobrescreve campo a campo."""
    with open(p, "r", encoding="utf-8-sig") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("extends", None)
    return _merge(_load_raw(p.parent / base), raw) if base else raw


def load_config(path) -> dict:
    p = Path(path).resolve()
    raw = _load_raw(p)
    cfg = _merge(DEFAULTS, raw)
    base = p.parent

    def res(x):
        if x is None:
            return None
        q = Path(x)
        return str(q if q.is_absolute() else (base / q).resolve())

    for c in cfg["cameras"]:
        c["video"] = res(c["video"])
    for k in ("timestamp_file", "framenum_file", "location_image"):
        cfg["metadata"][k] = res(cfg["metadata"].get(k))
    cfg["detector"]["model"] = res(cfg["detector"]["model"])
    if cfg["reid"].get("model"):
        cfg["reid"]["model"] = res(cfg["reid"]["model"])
    cfg["output"]["dir"] = res(cfg["output"]["dir"])
    if cfg["evaluation"].get("gt_dir"):
        cfg["evaluation"]["gt_dir"] = res(cfg["evaluation"]["gt_dir"])
    cal = cfg["spatial"].get("calibration")
    if cal:
        from .calibration import load_calibration
        cal["pattern"] = res(cal["pattern"].replace("{cam}", "@@CAM@@")).replace("@@CAM@@", "{cam}")
        for cam, H in load_calibration(cal, [c["id"] for c in cfg["cameras"]]).items():
            cfg["spatial"]["homographies"].setdefault(cam, H)
    if cfg["evaluation"].get("roi_pattern"):
        cfg["evaluation"]["roi_pattern"] = res(cfg["evaluation"]["roi_pattern"].replace("{cam}", "@@CAM@@")).replace("@@CAM@@", "{cam}")
    cfg["_config_path"] = str(p)
    return cfg
