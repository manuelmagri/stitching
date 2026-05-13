"""Matching FLANN tra due `FeatureSet`, con Lowe ratio + filtro RANSAC.

Catena standard per due frame consecutivi:

  1. FLANN-LSH knnMatch (k=2) sui descrittori ORB binari
  2. Lowe ratio test: tiene solo i match dove il migliore e' molto piu' vicino del secondo
  3. RANSAC su omografia per scartare gli outlier geometrici

L'omografia stessa NON e' un output principale (il VO usera' la matrice
essenziale), serve solo come filtro geometrico: per scene aeree a corto
baseline la planarita' del terreno la rende affidabile.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .feature import FeatureSet


# Parametri FLANN-LSH adatti ai descrittori ORB binari (32 byte).
# Stessi valori del vecchio prototipo: tornano un buon compromesso tra
# velocita' e richiamo su queste immagini.
_FLANN_INDEX_LSH = 6
_INDEX_PARAMS = dict(
    algorithm=_FLANN_INDEX_LSH,
    table_number=6,
    key_size=12,
    multi_probe_level=1,
)
_SEARCH_PARAMS = dict(checks=50)


@dataclass(frozen=True)
class MatchResult:
    """Esito del matching geometricamente filtrato tra due frame.

    `pts1`, `pts2` sono shape (N, 2) float32 e contengono SOLO gli inlier
    (post Lowe + post RANSAC), in coordinate pixel del rispettivo frame.
    `matches` sono i DMatch corrispondenti, comodi per `cv2.drawMatches`.
    `H` e' l'omografia stimata (3x3 float64) o None se non e' stata trovata.
    """

    pts1: np.ndarray            # (N, 2) float32, coordinate pixel frame 1
    pts2: np.ndarray            # (N, 2) float32, coordinate pixel frame 2
    matches: tuple              # tuple[cv2.DMatch], len = N
    H: np.ndarray | None

    def __len__(self) -> int:
        return len(self.matches)


def _new_flann() -> cv2.FlannBasedMatcher:
    return cv2.FlannBasedMatcher(_INDEX_PARAMS, _SEARCH_PARAMS)


def match_frames(fs1: FeatureSet,
                 fs2: FeatureSet,
                 ratio: float = 0.7,
                 ransac_thresh_px: float = 3.0,
                 min_matches: int = 10) -> MatchResult | None:
    """Match FLANN + Lowe + RANSAC tra due FeatureSet.

    Parameters
    ----------
    fs1, fs2 : FeatureSet
        Feature dei due frame (di solito i-1 e i). Le coordinate dei
        keypoint devono essere gia' nel sistema dell'immagine intera.
    ratio : float
        Soglia Lowe (0..1). Valori piu' alti -> piu' match ma piu' rumore.
        0.7 e' il default classico di Lowe; per scene aeree con tanta
        ripetizione (campi) puo' valere 0.6.
    ransac_thresh_px : float
        Soglia di reprojection error per RANSAC su omografia, in pixel.
    min_matches : int
        Numero minimo di match dopo Lowe per tentare RANSAC. Sotto questa
        soglia ritorna None (non ha senso fittare un'omografia).

    Returns
    -------
    MatchResult | None
        None se ci sono troppi pochi match per stimare la geometria; il
        chiamante deve gestire il fallback (skip frame o ricalibrare).
    """
    if fs1.descriptors.dtype != np.uint8 or fs2.descriptors.dtype != np.uint8:
        raise ValueError("FLANN-LSH richiede descrittori ORB uint8")
    if len(fs1) < 2 or len(fs2) < 2:
        return None

    flann = _new_flann()
    knn = flann.knnMatch(fs1.descriptors, fs2.descriptors, k=2)

    # Lowe ratio: alcune coppie possono avere meno di 2 vicini in zone
    # povere; le filtriamo prima del test per non sollevare IndexError.
    good: list[cv2.DMatch] = [
        pair[0] for pair in knn
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance
    ]
    if len(good) < min_matches:
        return None

    pts1 = np.float32([fs1.keypoints[m.queryIdx].pt for m in good])
    pts2 = np.float32([fs2.keypoints[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransac_thresh_px)
    if H is None or mask is None:
        return None

    inlier = mask.ravel().astype(bool)
    if inlier.sum() < min_matches:
        return None

    return MatchResult(
        pts1=pts1[inlier],
        pts2=pts2[inlier],
        matches=tuple(m for m, k in zip(good, inlier) if k),
        H=H,
    )


def draw_matches(bgr1: np.ndarray, fs1: FeatureSet,
                 bgr2: np.ndarray, fs2: FeatureSet,
                 result: MatchResult) -> np.ndarray:
    """Visualizza i match inlier su immagine affiancata (debug)."""
    return cv2.drawMatches(
        bgr1, fs1.keypoints,
        bgr2, fs2.keypoints,
        result.matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
