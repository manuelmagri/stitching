"""Estrazione feature ORB con suddivisione in nxn celle.

Le immagini aeree hanno spesso contrasto disomogeneo (vegetazione fitta vs.
strada vs. tetto): un singolo ORB su tutta l'immagine concentra i keypoint
nelle zone ad alto contrasto e lascia povere di feature le aree omogenee. La
griglia nxn forza una distribuzione spaziale piu' uniforme, migliorando la
qualita' del matching e la stabilita' di findEssentialMat / findHomography.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class FeatureSet:
    """Keypoints e descrittori di una singola immagine.

    `keypoints` e' una tupla per essere immutabile e hashable-friendly;
    `descriptors` e' un ndarray Nx32 uint8 (ORB). Le coordinate dei
    keypoint sono gia' nel sistema di riferimento dell'immagine intera.
    """

    keypoints: tuple
    descriptors: np.ndarray  # (N, 32) uint8

    def __len__(self) -> int:
        return len(self.keypoints)


def _orb_for_cell(features_per_cell: int) -> cv2.ORB:
    # `fastThreshold` basso aiuta a trovare punti anche in celle a basso
    # contrasto (campi, asfalto uniforme); `edgeThreshold` ridotto evita
    # di scartare punti vicini al bordo delle celle.
    return cv2.ORB_create(
        nfeatures=features_per_cell,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=10,
    )


def detect_orb_nxn(gray: np.ndarray,
                   n: int,
                   features_per_cell: int = 300) -> FeatureSet:
    """Estrae ORB suddividendo `gray` in una griglia nxn.

    Parameters
    ----------
    gray : np.ndarray
        Immagine in scala di grigi (H, W), uint8.
    n : int
        Numero di celle per lato. Il valore tipico cresce con la dimensione
        del mosaico: 5 per i primi frame, 7-8 quando il canvas diventa grande.
    features_per_cell : int
        Target di feature ORB per cella. Il totale finale puo' essere
        inferiore se alcune celle sono povere di angoli.

    Returns
    -------
    FeatureSet
        Keypoints (con coordinate globali) e descrittori concatenati.

    Raises
    ------
    ValueError
        Se `gray` non e' 2D uint8 o se `n < 1`.
    RuntimeError
        Se nessuna cella produce descrittori.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError(
            f"Atteso grayscale uint8 (H, W); ricevuto shape={gray.shape}, "
            f"dtype={gray.dtype}. Converti con cv2.cvtColor(..., COLOR_BGR2GRAY)."
        )
    if n < 1:
        raise ValueError(f"n deve essere >= 1, ricevuto {n}")

    H, W = gray.shape
    orb = _orb_for_cell(features_per_cell)

    all_kps: list[cv2.KeyPoint] = []
    desc_blocks: list[np.ndarray] = []

    # np.linspace con dtype int da' bordi cella che coprono esattamente
    # [0, W) senza buchi ne' sovrapposizioni, anche quando W non e'
    # divisibile per n.
    xs = np.linspace(0, W, n + 1, dtype=np.int32)
    ys = np.linspace(0, H, n + 1, dtype=np.int32)

    for r in range(n):
        for c in range(n):
            y0, y1 = int(ys[r]), int(ys[r + 1])
            x0, x1 = int(xs[c]), int(xs[c + 1])
            cell = gray[y0:y1, x0:x1]
            if cell.size == 0:
                continue

            kps, des = orb.detectAndCompute(cell, None)
            if des is None or len(kps) == 0:
                continue

            # Riporta i keypoint dal frame della cella a quello dell'immagine.
            for kp in kps:
                kp.pt = (kp.pt[0] + x0, kp.pt[1] + y0)
            all_kps.extend(kps)
            desc_blocks.append(des)

    if not desc_blocks:
        raise RuntimeError(
            f"Nessuna feature ORB estratta nelle {n*n} celle. "
            "Immagine troppo uniforme o celle troppo piccole."
        )

    descriptors = np.vstack(desc_blocks)
    return FeatureSet(keypoints=tuple(all_kps), descriptors=descriptors)


def draw_keypoints(bgr: np.ndarray, features: FeatureSet) -> np.ndarray:
    """Overlay dei keypoint per debug visivo. Non modifica `bgr`."""
    return cv2.drawKeypoints(
        bgr, features.keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
    )
