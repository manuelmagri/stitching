"""Matching ORB, stima similarity 2D inlier-only e costruzione coppie leg-based.

La selezione delle coppie da matchare ha due componenti:
1. Catena intra-leg (`build_leg_pairs`): coppie consecutive e skip-one nella sequenza
   estesa bridge_prev + frame_tenuti + bridge_next. I bridge condivisi fra leg adiacenti
   chiudono la catena a passare attraverso la curva.
2. Aggancio cross-leg laterale (`build_cross_leg_pairs`): per ogni frame del mosaico,
   coppie dirette con i frame piu' prossimi di OTHER leg, entro un raggio derivato
   dall'overlap laterale. Risolve i disalineamenti fra passate (la sola catena via
   bridge curva non basta perche' quei frame sono visualmente sgualciti dal roll/yaw rate).
"""
import cv2
import numpy as np


def make_matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def build_leg_pairs(
    legs: list[dict],
    stitch_indices_per_leg: list[list[int]],
) -> list[tuple[int, int]]:
    """Coppie (a, b) di indici (in `records`) da matchare.

    Per ogni leg costruisce l'extended sequence (bridge_prev + frame_tenuti + bridge_next)
    e aggiunge le coppie consecutive e skip-one. I duplicati fra leg adiacenti (gli stessi
    bridge compaiono in due leg) vengono deduplicati.
    """
    pairs_set: set[tuple[int, int]] = set()
    for leg, kept in zip(legs, stitch_indices_per_leg):
        ext = list(leg["bridge_prev"]) + list(kept) + list(leg["bridge_next"])
        n = len(ext)
        for offset in (1, 2):
            for i in range(n - offset):
                a, b = ext[i], ext[i + offset]
                if a == b:
                    continue
                pair = (a, b) if a < b else (b, a)
                pairs_set.add(pair)
    return sorted(pairs_set)


def cross_leg_radius_m(
    image_width_px: int,
    gsd_m_per_px: float,
    lateral_overlap: float,
    margin: float = 1.4,
) -> float:
    """Raggio entro il quale cercare frame di leg adiacenti per il matching cross-leg.

    Due frame su leg adiacenti con `lateral_overlap` hanno centri a distanza
    cross-track ~ w_px * gsd * (1 - lateral_overlap). Il margine cattura anche
    frame leggermente sfasati in along-track.
    """
    footprint_cross_m = image_width_px * gsd_m_per_px
    return footprint_cross_m * max(1.0 - lateral_overlap, 0.05) * margin


def build_cross_leg_pairs(
    stitch_indices_per_leg: list[list[int]],
    positions_m: list[tuple[float, float]],
    radius_m: float,
    max_neighbors_per_frame: int = 4,
) -> list[tuple[int, int]]:
    """Coppie (a, b) fra frame del mosaico appartenenti a leg DIVERSI entro radius_m.

    Per ogni frame mosaico, prende fino a max_neighbors_per_frame frame piu' vicini
    in GPS che appartengono a un leg differente. I duplicati (i, j) e (j, i) sono
    deduplicati.

    I bridge curva NON entrano qui: vengono gia' agganciati dalla catena di
    `build_leg_pairs`. Qui vogliamo solo connessioni dirette fra le passate dritte,
    dove le immagini sono nitide e le feature ORB matchano bene.
    """
    if len(stitch_indices_per_leg) < 2:
        return []

    frame_leg: dict[int, int] = {}
    for leg_id, kept in enumerate(stitch_indices_per_leg):
        for f in kept:
            frame_leg[f] = leg_id

    frames = sorted(frame_leg.keys())
    if not frames:
        return []
    pts = np.array([positions_m[f] for f in frames], dtype=np.float32)

    pairs_set: set[tuple[int, int]] = set()
    for i, fi in enumerate(frames):
        d = np.hypot(pts[:, 0] - pts[i, 0], pts[:, 1] - pts[i, 1])
        d[i] = np.inf
        leg_i = frame_leg[fi]
        # Solo frame entro raggio E appartenenti a leg diversi
        mask = (d < radius_m) & np.array(
            [frame_leg[f] != leg_i for f in frames], dtype=bool
        )
        cand = np.where(mask)[0]
        if len(cand) > max_neighbors_per_frame:
            order = np.argsort(d[cand])[:max_neighbors_per_frame]
            cand = cand[order]
        for c in cand:
            fj = frames[int(c)]
            a, b = (fi, fj) if fi < fj else (fj, fi)
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
