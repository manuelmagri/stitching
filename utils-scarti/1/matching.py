"""Matching FLANN-LSH (per descrittori ORB binari) con Lowe ratio test e RANSAC."""
from __future__ import annotations

import cv2
import numpy as np


def make_flann() -> cv2.FlannBasedMatcher:
    """FLANN configurato per descrittori binari (ORB)."""
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,
        key_size=12,
        multi_probe_level=1,
    )
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


def match_descriptors(
    flann: cv2.FlannBasedMatcher,
    des1: np.ndarray | None,
    des2: np.ndarray | None,
    lowe_ratio: float = 0.7,
    min_matches: int = 12,
) -> list[cv2.DMatch]:
    """knnMatch + Lowe ratio test. Ritorna [] se ci sono troppo pochi match."""
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    pairs = flann.knnMatch(des1, des2, k=2)
    good: list[cv2.DMatch] = []
    for pp in pairs:
        if len(pp) < 2:
            continue
        m, n = pp
        if m.distance < lowe_ratio * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return []
    return good


def matched_points(
    kp1: list[cv2.KeyPoint],
    kp2: list[cv2.KeyPoint],
    matches: list[cv2.DMatch],
) -> tuple[np.ndarray, np.ndarray]:
    """Estrae le coordinate (N, 2) dei keypoint per ogni match."""
    q1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    q2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return q1, q2


def homography_ransac(
    q1: np.ndarray, q2: np.ndarray, ransac_thresh: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stima omografia q1 -> q2 con RANSAC. Ritorna (H, mask_bool)."""
    if len(q1) < 4 or len(q2) < 4:
        return None, None
    H, mask = cv2.findHomography(q1, q2, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return H, None
    return H, mask.ravel().astype(bool)


def essential_ransac(
    q1: np.ndarray, q2: np.ndarray, K: np.ndarray,
    prob: float = 0.999, thresh: float = 0.5,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Stima la matrice essenziale E con RANSAC. Ritorna (E, mask_bool)."""
    if len(q1) < 5 or len(q2) < 5:
        return None, None
    E, mask = cv2.findEssentialMat(q1, q2, K, method=cv2.RANSAC, prob=prob, threshold=thresh)
    if mask is None:
        return E, None
    return E, mask.ravel().astype(bool)
