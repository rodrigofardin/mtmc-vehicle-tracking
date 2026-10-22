"""Calibração do dataset (AIC22 CityFlowV2): homografia por câmera para um plano de chão comum.

Formato verificado (`<subset>/<cam>/calibration.txt`, 1ª linha):
    Homography matrix: a b c;d e f;g h 1
Verificação empírica: `inv(H) @ (x, y, 1)` leva pixel -> (latitude, longitude) e cai no centro GPS do cenário
informado no ReadMe do dataset (S01: 42.525678, -90.723601), enquanto `H @` pixel não. O erro de
reprojeção reportado no arquivo é de 5–11 px (calibração manual): trate o cue espacial como aproximado.

Aqui o plano é convertido para metros locais (leste, norte) em torno de `ref_latlon`, para que
`spatial.max_ground_dist` seja em metros.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_aicity_homography(path) -> np.ndarray:
    line = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    if ":" not in line:
        raise ValueError(f"{path}: 1ª linha sem 'Homography matrix:'")
    rows = [[float(v) for v in r.split()] for r in line.split(":", 1)[1].strip().split(";")]
    H = np.array(rows, dtype=float)
    if H.shape != (3, 3):
        raise ValueError(f"{path}: homografia deve ser 3x3, obtido {H.shape}")
    return H


def image_to_local_meters(H_gps_to_img: np.ndarray, ref_latlon) -> List[List[float]]:
    """Matriz 3x3: pixel (x, y, 1) -> (leste_m, norte_m, 1) relativa a `ref_latlon`."""
    lat0, lon0 = ref_latlon
    ky = 110540.0                                   # m por grau de latitude (aprox.)
    kx = 111320.0 * math.cos(math.radians(lat0))    # m por grau de longitude
    G = np.linalg.inv(H_gps_to_img)                 # pixel -> (lat, lon)
    A = np.array([[0.0, kx, -lon0 * kx], [ky, 0.0, -lat0 * ky], [0.0, 0.0, 1.0]])
    return (A @ G).tolist()


def load_calibration(spec: dict, cam_ids) -> Dict[str, List[List[float]]]:
    """`spec`: {pattern: ".../{cam}/calibration.txt", ref_latlon: [lat, lon]}; câmeras sem arquivo são omitidas."""
    out = {}
    for c in cam_ids:
        p = Path(spec["pattern"].format(cam=c))
        if p.is_file():
            out[c] = image_to_local_meters(parse_aicity_homography(p), spec["ref_latlon"])
    return out
