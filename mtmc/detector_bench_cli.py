"""python -m mtmc.detector_bench_cli --config configs/detector_bench.yaml --only yolov8n,yolov10l [--stride 10]

Cada modelo gera uma entrada em `<out>/results.json` (retomável). Execute um modelo por processo para
liberar VRAM entre eles (o script já faz `del` + `empty_cache`, mas processos separados são mais confiáveis).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import yaml

from .detector_bench import metrics, run_detector, sample_frames, vram_used_mb

log = logging.getLogger("bench")


def resolve_weights(w: str, models_dir: Path) -> str:
    p = Path(w)
    if p.is_file():
        return str(p)
    if (models_dir / w).is_file():
        return str(models_dir / w)
    from ultralytics.utils.downloads import attempt_download_asset
    cwd = os.getcwd()
    os.chdir(models_dir)                       # downloads do Ultralytics caem no diretório atual
    try:
        got = attempt_download_asset(w)
    finally:
        os.chdir(cwd)
    return str(models_dir / Path(got).name)


def build(spec: dict, backend: str, models_dir: Path):
    kind = spec["kind"]
    if kind == "ultralytics":
        from .detectors import UltralyticsDetector
        w = resolve_weights(spec["weights"], models_dir)
        hw = spec.get("imgsz", [544, 960])
        if backend == "trt":
            from .detector_export import export_ultralytics_trt
            eng = export_ultralytics_trt(w, hw, models_dir / "engines")
            return UltralyticsDetector(eng, imgsz=hw, name=f"{spec['name']}-trt-fp16", half=False), str(eng)
        return UltralyticsDetector(w, imgsz=max(hw), half=True, name=f"{spec['name']}-pt-fp16"), w
    if kind in ("rfdetr", "dfine"):
        from .detector_export_detr import build_detr_detector
        return build_detr_detector(spec, backend, models_dir)
    raise ValueError(f"kind desconhecido: {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detector_bench.yaml")
    ap.add_argument("--only", default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--backends", default=None, help="sobrescreve: pt,trt")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    root = Path(cfg["dataset_root"])
    models_dir = Path(cfg.get("models_dir", "models"))
    out_dir = Path(cfg.get("out_dir", "outputs/detector_bench"))
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / "results.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.is_file() else {}
    stride = a.stride or cfg.get("stride", 10)
    samples = sample_frames(root, cfg.get("scenes", ["S01", "S02"]), stride)
    log.info("%d câmeras, %d frames amostrados (stride %d)", len(samples), sum(len(s[2]) for s in samples), stride)
    only = set(a.only.split(",")) if a.only else None
    import torch
    for spec in cfg["models"]:
        if only and spec["name"] not in only:
            continue
        for backend in (a.backends.split(",") if a.backends else spec.get("backends", ["trt"])):
            key = f"{spec['name']}|{backend}"
            if key in results and not only:
                continue
            log.info("=== %s (%s)", spec["name"], backend)
            try:
                v0 = vram_used_mb()
                t0 = time.time()
                det, art = build(spec, backend, models_dir)
                load_s = time.time() - t0
                det.detect(__import__("numpy").zeros((1080, 1920, 3), "uint8"), 0.05)      # aquece/aloca
                recs, lat = run_detector(det, samples)
                m = metrics(recs, lat)
                m.update(model=spec["name"], backend=backend, artifact=art, imgsz=spec.get("imgsz"),
                         vram_mb=round(vram_used_mb() - v0, 0), build_load_s=round(load_s, 1), stride=stride)
                results[key] = m
                log.info("%s: recall@.5/c.25=%.3f  AP50p=%.3f  p50=%.1f ms  vram=%d MB", key, m["recall@0.5_conf0.25"],
                         m["AP50_partial"], m.get("latency_ms_p50", -1), m["vram_mb"])
            except Exception as e:  # um modelo que falha não derruba os demais
                log.exception("falhou: %s", key)
                results[key] = {"model": spec["name"], "backend": backend, "error": f"{type(e).__name__}: {e}"}
            res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            del det
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
