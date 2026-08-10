"""Corrispondenze fra descrittori e stima della similarita' che lega due scatti.

Quale coppia tentare lo decide `utils.pairing` dalla geometria; qui si lavora solo sulle
immagini. La coppia serve a stimare una trasformazione, non a produrre un incollaggio
intermedio: il mosaico e' una fase separata, a valle delle pose.

Sulle passate adiacenti di una greca le immagini sono capovolte l'una rispetto all'altra,
perche' le rotte differiscono di 180 gradi. I descrittori ORB sono invarianti per
rotazione e reggono, ma il tasso di inlier cala: proprio quelle coppie sono, senza GPS,
le uniche chiusure d'anello fra le passate. Da qui `agrees_with_seed`, che confronta la
similarita' stimata con quella prevista dalle pose iniziali e permette di scartare subito
le stime assurde, prima che entrino nell'ottimizzazione.
"""
import numpy as np

import cv2


def make_matcher() -> cv2.BFMatcher:
    return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


def match_descriptors(
    matcher: cv2.BFMatcher,
    des1: np.ndarray,
    des2: np.ndarray,
    ratio: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Indici delle corrispondenze che superano il test del rapporto di Lowe.

    Ritorna due array di pari lunghezza: posizioni in `des1` e posizioni in `des2`.
    """
    if des1 is None or des2 is None or len(des1) < 8 or len(des2) < 8:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    coppie = matcher.knnMatch(des1, des2, k=2)
    query, train = [], []
    for gruppo in coppie:
        if len(gruppo) < 2:
            continue
        m, n = gruppo
        if m.distance < ratio * n.distance:
            query.append(m.queryIdx)
            train.append(m.trainIdx)
    return np.array(query, dtype=int), np.array(train, dtype=int)


def similarity_from_matches(
    points1: np.ndarray,
    points2: np.ndarray,
    idx1: np.ndarray,
    idx2: np.ndarray,
    ransac_thresh: float = 3.0,
    min_matches: int = 10,
    min_inliers: int = 12,
) -> tuple[np.ndarray | None, int]:
    """Similarita' 3x3 che porta i pixel della prima immagine sulla seconda, e n. inlier.

    Restituisce (None, n) quando le corrispondenze superstiti sono troppo poche perche' la
    stima significhi qualcosa: e' il modulo a sapere quanti inlier gli servono, non chi lo
    chiama.
    """
    if len(idx1) < min_matches:
        return None, 0

    p1 = points1[idx1].astype(np.float32)
    p2 = points2[idx2].astype(np.float32)
    A, mask = cv2.estimateAffinePartial2D(
        p1, p2, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh, maxIters=2000
    )
    if A is None or mask is None:
        return None, 0

    n_inlier = int(mask.sum())
    if n_inlier < min_inliers:
        return None, n_inlier

    H = np.eye(3, dtype=np.float64)
    H[:2, :] = A
    return H, n_inlier


def seed_disagreement_px(
    H: np.ndarray,
    M_i: np.ndarray,
    M_j: np.ndarray,
    image_size: tuple[int, int],
) -> float:
    """Massimo scarto, in pixel, fra la similarita' stimata e quella prevista dal seed.

    Le pose iniziali prevedono che i pixel di i finiscano su j attraverso `inv(M_j) @ M_i`.
    Se la stima fotografica se ne discosta di molto piu' dell'incertezza del seed, si
    tratta quasi sempre di un aggancio su tessitura ripetitiva, e vale la pena scartarla:
    un vincolo sbagliato pesato con la radice dei suoi inlier fa piu' danno di un vincolo
    mancante.
    """
    w, h = image_size
    corners = np.array(
        [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]], dtype=np.float64
    )
    previsto = np.linalg.inv(M_j) @ M_i
    a = (corners @ np.asarray(H, dtype=np.float64).T)[:, :2]
    b = (corners @ previsto.T)[:, :2]
    return float(np.linalg.norm(a - b, axis=1).max())


def agrees_with_seed(
    H: np.ndarray,
    M_i: np.ndarray,
    M_j: np.ndarray,
    image_size: tuple[int, int],
    tolerance: float = 0.25,
) -> bool:
    """Se la similarita' stimata sia compatibile con le pose iniziali.

    `tolerance` e' espressa in frazioni della diagonale dell'immagine, cosi' vale a
    qualunque risoluzione. Il valore di default e' largo di proposito: serve a scartare gli
    agganci assurdi su tessitura ripetitiva, non a imporre il seed alle immagini, che sono
    la misura piu' precisa che abbiamo.
    """
    w, h = image_size
    return seed_disagreement_px(H, M_i, M_j, image_size) <= tolerance * float(np.hypot(w, h))
