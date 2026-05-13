"""Feature matching FLANN + Lowe ratio test (consumati dalla VO)."""
import cv2
import numpy as np

from .timing import timed


@timed
def make_flann_matcher():
    """Costruisce un FLANN matcher configurato per descriptors ORB (binari, LSH)."""
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)


@timed
def match_pairs(matcher, kp1, des1, kp2, des2, ratio=0.7, min_good=10):
    """Match knn=2 con Lowe's ratio test.

    Ritorna (good_matches, q1_pts, q2_pts) come array float32 di forma (N, 2).
    Ritorna (None, None, None) se i descriptors mancano o se i match validi sono meno di `min_good`.
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return None, None, None

    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        # FLANN può restituire meno di 2 vicini ai bordi del dataset
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < min_good:
        return None, None, None

    q1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    q2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return good, q1, q2
