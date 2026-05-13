"""Detection di feature ORB con suddivisione dell'immagine in griglia n x n.

Distribuire la detection in celle evita che le feature si concentrino in una
zona dell'immagine: importante per stimare omografie e matrici essenziali
robuste su panorami nadir.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def detect_orb_grid(
    gray: np.ndarray,
    n_grid: int,
    n_features_total: int,
    save_path: Path | str | None = None,
) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
    """Detect+compute ORB su una griglia n x n e ricompone keypoint+descriptors.

    `n_features_total` e' il budget complessivo per tutta l'immagine; viene
    diviso uniformemente tra le celle (con un minimo di 50 per cella).

    Se `save_path` e' fornito, salva un'immagine di debug con i keypoint disegnati.
    """
    h, w = gray.shape[:2]
    per_cell = max(50, int(np.ceil(n_features_total / max(1, n_grid * n_grid))))
    orb = cv2.ORB_create(nfeatures=per_cell)

    cell_h = h // n_grid
    cell_w = w // n_grid
    all_kp: list[cv2.KeyPoint] = []
    all_des: list[np.ndarray] = []
    for r in range(n_grid):
        for c in range(n_grid):
            y0 = r * cell_h
            y1 = (r + 1) * cell_h if r < n_grid - 1 else h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w if c < n_grid - 1 else w
            patch = gray[y0:y1, x0:x1]
            kps, des = orb.detectAndCompute(patch, None)
            if des is None or len(kps) == 0:
                continue
            for k in kps:
                k.pt = (k.pt[0] + x0, k.pt[1] + y0)
            all_kp.extend(kps)
            all_des.append(des)

    if not all_des:
        return [], None
    descriptors = np.vstack(all_des)
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        img = cv2.drawKeypoints(gray, all_kp, None, color=(0, 255, 0), flags=0)
        cv2.imwrite(str(save_path), img)
    return all_kp, descriptors
