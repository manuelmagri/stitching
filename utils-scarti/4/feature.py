"""Feature ORB con suddivisione in nxn quadranti.

OLD-CODE estraeva ORB sull'intera immagine, accumulando feature concentrate
in zone ad alto contrasto. Suddividere in quadranti e applicare un budget
per cella forza una distribuzione spaziale piu' uniforme: utile per la
matrice essenziale (parallasse meglio condizionato) e per l'omografia (RANSAC
con punti distribuiti e' molto piu' stabile).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FeatureSet:
    """Wrapper minimale: keypoints OpenCV + descrittori np.uint8."""
    keypoints: tuple
    descriptors: np.ndarray | None

    def __len__(self) -> int:
        return 0 if self.descriptors is None else len(self.descriptors)


def _split_quadrants(img: np.ndarray, n: int):
    """Genera (y0, y1, x0, x1) per ognuno degli n*n quadranti dell'immagine.

    L'ultimo quadrante per riga/colonna assorbe i pixel di resto, evitando
    bordi neri se h o w non sono multipli di n.
    """
    h, w = img.shape[:2]
    ys = [(i * h) // n for i in range(n + 1)]
    xs = [(j * w) // n for j in range(n + 1)]
    for i in range(n):
        for j in range(n):
            yield ys[i], ys[i + 1], xs[j], xs[j + 1]


def detect_orb_quadrants(img: np.ndarray,
                         n: int = 5,
                         features_per_cell: int = 200,
                         orb_params: dict | None = None) -> FeatureSet:
    """Rileva ORB suddividendo l'immagine in n*n quadranti.

    `features_per_cell` mette un cap su quante feature al massimo si
    estraggono da ciascun quadrante (budget uniforme). Il budget totale
    teorico e' n*n*features_per_cell, ma le celle povere ne producono meno.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    base_params = dict(
        nfeatures=features_per_cell,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=12,
    )
    if orb_params:
        base_params.update(orb_params)
    orb = cv2.ORB_create(**base_params)

    all_kp: list = []
    all_des: list[np.ndarray] = []

    for y0, y1, x0, x1 in _split_quadrants(gray, n):
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        kp, des = orb.detectAndCompute(patch, None)
        if des is None or len(kp) == 0:
            continue
        # Riporta i keypoint in coordinate immagine intera.
        for k in kp:
            k.pt = (k.pt[0] + x0, k.pt[1] + y0)
        all_kp.extend(kp)
        all_des.append(des)

    if not all_des:
        return FeatureSet(keypoints=tuple(), descriptors=None)

    descriptors = np.vstack(all_des)
    return FeatureSet(keypoints=tuple(all_kp), descriptors=descriptors)


def disegna_feature(img: np.ndarray, fs: FeatureSet, output_path: str | None = None):
    """Disegna i keypoint sull'immagine. Se output_path e' fornito, salva su disco."""
    vis = cv2.drawKeypoints(
        img, fs.keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
    if output_path is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        cv2.imwrite(output_path, vis)
    return vis
