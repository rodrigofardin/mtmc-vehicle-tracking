"""Monta o conjunto de recortes de Re-ID a partir do GT do CityFlowV2 (cenários de TREINO escolhidos).

    python -m reid_training.build_dataset --root /AICity22_Track1_MTMC_Tracking \\
        --scenes train/S03 train/S04 validation/S05 --out reid_training/data/train

Cada veículo (ID global do GT) vira uma classe `<cenário>_<id>`; os recortes vêm de várias câmeras (positivos
entre câmeras). S01 e S02 NÃO devem estar em `--scenes`: são reservados para avaliação (os veículos são distintos
entre cenários, então não há vazamento de identidade).
Saída: `<out>/<cenário>/<id>/<cam>_<frame>.jpg` + `<out>/manifest.csv` (path,label,cam,scene,frame).
"""
from __future__ import annotations

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from mtmc.evaluation import load_gt

log = logging.getLogger("reid_data")


def pick_even(obs, k):
    if len(obs) <= k:
        return obs
    idx = np.linspace(0, len(obs) - 1, k).round().astype(int)
    return [obs[i] for i in sorted(set(idx))]


def build_scene(root: Path, scene: str, out: Path, per_track: int, min_side: int, writer) -> dict:
    sdir = root / scene
    cams = sorted(p.name for p in sdir.iterdir() if p.is_dir() and p.name.startswith("c"))
    gt = load_gt(sdir, cams)
    name = scene.split("/")[-1]
    stats = {"cams": 0, "crops": 0, "ids": set()}
    for cam in cams:
        if cam not in gt:
            continue
        video = sdir / cam / "vdo.avi"
        cap = cv2.VideoCapture(str(video))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        tracks = defaultdict(list)
        for f, items in gt[cam].items():
            for gid, b in items:
                w, h = b[2] - b[0], b[3] - b[1]
                inside = b[0] > 2 and b[1] > 2 and b[2] < W - 2 and b[3] < H - 2   # caixas cortadas pela borda: embedding ruim
                if w >= min_side and h >= min_side and inside and 0.3 < w / h < 4.0:
                    tracks[gid].append((f, b))
        want = defaultdict(list)
        for gid, obs in tracks.items():
            obs.sort()
            for f, b in pick_even(obs, per_track):
                want[f].append((gid, b))
        if not want:
            cap.release()
            continue
        stats["cams"] += 1
        for f in range(max(want) + 1):
            if f not in want:
                if not cap.grab():
                    break
                continue
            ok, img = cap.read()
            if not ok:
                break
            for gid, b in want[f]:
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                crop = img[max(y1, 0):y2, max(x1, 0):x2]
                if crop.size == 0:
                    continue
                d = out / name / str(gid)
                d.mkdir(parents=True, exist_ok=True)
                p = d / f"{cam}_{f:05d}.jpg"
                cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                writer.writerow([str(p.relative_to(out)).replace("\\", "/"), f"{name}_{gid}", cam, name, f])
                stats["crops"] += 1
                stats["ids"].add(gid)
        cap.release()
        log.info("%s/%s: %d recortes acumulados", name, cam, stats["crops"])
    stats["ids"] = len(stats["ids"])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True, help="ex.: train/S03 train/S04 validation/S05")
    ap.add_argument("--out", default="reid_training/data/train")
    ap.add_argument("--per-track", type=int, default=25, help="recortes por (câmera, veículo)")
    ap.add_argument("--min-side", type=int, default=48)
    ap.add_argument("--eval-set", action="store_true",
                    help="conjunto de AVALIAÇÃO: aceita só S01/S02 (nunca use para treinar)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    names = {s.split("/")[-1] for s in a.scenes}
    if a.eval_set:
        if not names <= {"S01", "S02"}:
            raise SystemExit("--eval-set só aceita S01/S02")
    elif names & {"S01", "S02"}:
        raise SystemExit(f"{sorted(names & {'S01', 'S02'})} são reservados para avaliação; remova de --scenes (ou use --eval-set)")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "cam", "scene", "frame"])
        for s in a.scenes:
            st = build_scene(Path(a.root), s, out, a.per_track, a.min_side, w)
            log.info("== %s: %d câmeras, %d veículos, %d recortes", s, st["cams"], st["ids"], st["crops"])


if __name__ == "__main__":
    main()
