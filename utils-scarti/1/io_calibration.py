"""Caricamento dei parametri di calibrazione della camera dal disco."""
import json
from pathlib import Path

import numpy as np


def load_calibration(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Carica K (3x3), P (3x4), distortion (5,) da `data/`.

    Le immagini in `immagini_senza_distorsione/` sono gia' undistorte: la distortion
    serve solo per riferimento o per rifare l'undistort sui raw.
    """
    with open(data_dir / "calibration.txt") as f:
        d = json.load(f)
    K = np.array(d["cameraMatrix"], dtype=np.float64)
    P = np.array(d["projMatrix"], dtype=np.float64)

    dist = None
    dist_path = data_dir / "dist.txt"
    if dist_path.exists():
        with open(dist_path) as f:
            dist = np.array(json.load(f), dtype=np.float64)
    return K, P, dist
