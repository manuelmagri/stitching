"""Matching di feature ORB tra due frame: FLANN-LSH + Lowe ratio + RANSAC.

Output principale: coppie di punti (q1, q2) gia' filtrate dagli inlier
RANSAC dell'omografia, pronte sia per il warp del mosaico sia per la
matrice essenziale della VO.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np

from .feature import FeatureSet


@dataclass
class MatchResult:
    """Esito del matching tra due frame."""
    q1: np.ndarray            # (N, 2) punti nell'immagine 1, float32
    q2: np.ndarray            # (N, 2) punti nell'immagine 2, float32
    H: np.ndarray | None      # omografia 3x3 stimata da RANSAC (immagine1 -> immagine2)
    inlier_matches: list      # cv2.DMatch sopravvissuti a Lowe + RANSAC
    raw_match_count: int      # numero di match prima dei filtri (debug)

    def __len__(self) -> int:
        return len(self.q1)


def _flann_lsh():
    """Matcher FLANN per descrittori binari ORB (LSH)."""
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


def match_features(fs1: FeatureSet,
                   fs2: FeatureSet,
                   ratio: float = 0.75,
                   ransac_thresh_px: float = 3.0,
                   min_matches: int = 12) -> MatchResult | None:
    """Match tra due FeatureSet.

    Pipeline:
      1. knnMatch (k=2) con FLANN-LSH
      2. Lowe ratio test (default 0.75 -> piu' permissivo del 0.7 di OLD-CODE,
         compensato dal RANSAC successivo)
      3. cv2.findHomography con RANSAC, mantieni solo gli inlier

    Ritorna `None` se non ci sono abbastanza match per stimare un'omografia.
    """
    if len(fs1) == 0 or len(fs2) == 0:
        return None

    flann = _flann_lsh()
    knn = flann.knnMatch(fs1.descriptors, fs2.descriptors, k=2)

    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    raw = len(good)
    if raw < min_matches:
        return None

    q1 = np.float32([fs1.keypoints[m.queryIdx].pt for m in good])
    q2 = np.float32([fs2.keypoints[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(q1, q2, cv2.RANSAC, ransac_thresh_px)
    if H is None or mask is None:
        return None

    mask_bool = mask.ravel().astype(bool)
    if mask_bool.sum() < min_matches:
        return None

    return MatchResult(
        q1=q1[mask_bool],
        q2=q2[mask_bool],
        H=H,
        inlier_matches=[good[i] for i in range(len(good)) if mask_bool[i]],
        raw_match_count=raw,
    )


def disegna_match(img1: np.ndarray, fs1: FeatureSet,
                  img2: np.ndarray, fs2: FeatureSet,
                  result: MatchResult,
                  output_path: str):
    """Salva un'immagine con i match inlier disegnati tra img1 e img2."""
    vis = cv2.drawMatches(
        img1, fs1.keypoints,
        img2, fs2.keypoints,
        result.inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    cv2.imwrite(output_path, vis)
    return output_path
