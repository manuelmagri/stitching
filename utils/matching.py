"""Coppie di vicini da GPS, matching ORB e stima similarity 2D inlier-only."""
import cv2
import numpy as np


def make_matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def neighbor_radius_m(footprint_m: float, lateral_overlap: float, frontal_overlap: float) -> float:
    """Raggio per filtrare i vicini in base all'overlap.

    Due frame il cui contenuto si sovrappone hanno centro a distanza < footprint*(1-min_overlap).
    Aggiungiamo margine per catturare anche coppie inter-leg (overlap laterale)."""
    min_overlap = max(0.0, min(lateral_overlap, frontal_overlap))
    return footprint_m * max(1.0 - min_overlap, 0.1) * 1.6


def build_neighbor_pairs(
    positions_m: list[tuple[float, float]],
    radius_m: float,
    max_neighbors_per_frame: int = 8,
) -> list[tuple[int, int]]:
    """Coppie (i, j) con i < j tali che distanza GPS < radius_m, max_neighbors_per_frame
    per ogni frame (i vicini piu' prossimi)."""
    n = len(positions_m)
    pts = np.array(positions_m, dtype=np.float32)
    pairs_set: set[tuple[int, int]] = set()

    for i in range(n):
        dists = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        dists[i] = np.inf  # escludi self
        candidates = np.where(dists < radius_m)[0]
        if len(candidates) > max_neighbors_per_frame:
            order = np.argsort(dists[candidates])[:max_neighbors_per_frame]
            candidates = candidates[order]
        for j in candidates:
            a, b = (i, int(j)) if i < j else (int(j), i)
            pairs_set.add((a, b))

    return sorted(pairs_set)


def match_descriptors(
    matcher: cv2.BFMatcher,
    des1: np.ndarray | None,
    des2: np.ndarray | None,
    ratio: float = 0.75,
) -> list:
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return []
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def similarity_from_matches(
    kp1, kp2, matches, ransac_thresh: float = 3.0
) -> tuple[np.ndarray | None, int]:
    """Similarity 3x3 che mappa pixel_kp1 -> pixel_kp2. Ritorna (H, n_inliers)."""
    if len(matches) < 10:
        return None, 0
    p1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    p2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    A, mask = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000
    )
    if A is None or mask is None:
        return None, 0
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A
    return H, int(mask.sum())
