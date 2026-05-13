"""Visual Odometry frame-a-frame con matrice essenziale + IMU scale.

Per ogni coppia di immagini consecutive:
    1. detection ORB griglia + match FLANN-LSH + Lowe ratio
    2. cv2.findEssentialMat (RANSAC) -> recoverPose
    3. la traslazione t (norma 1) viene moltiplicata per scales[i] (m) IMU
    4. la posa cumulativa viene aggiornata: T_world_i = T_world_{i-1} @ T_local
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .features import detect_orb_grid
from .matching import (
    essential_ransac,
    make_flann,
    match_descriptors,
    matched_points,
)


@dataclass
class VOResult:
    poses: list[np.ndarray]   # lista di matrici 4x4 in coordinate "VO world"
    path: np.ndarray          # (N, 3) array delle posizioni cumulative
    skipped: list[int]        # indici dei frame in cui il VO non e' riuscito


def run_vo(
    images_gray: list[np.ndarray],
    K: np.ndarray,
    scales: np.ndarray | None,
    *,
    n_grid: int = 5,
    n_features_total: int = 4000,
    lowe_ratio: float = 0.7,
    min_matches: int = 12,
    essential_prob: float = 0.999,
    essential_thresh: float = 0.5,
    use_imu_scales: bool = True,
    verbose: bool = True,
) -> VOResult:
    """Esegue VO sequenziale su una lista di immagini grayscale.

    `scales` deve contenere len(images) valori, dove `scales[i]` rappresenta la
    distanza percorsa tra il frame i-1 e il frame i (m). Per i=0 il valore non
    viene usato. Se `scales` e' None oppure `use_imu_scales=False`, la scala
    rimane unitaria e il VO produce un percorso adimensionale.
    """
    flann = make_flann()
    cur_pose = np.eye(4)
    poses: list[np.ndarray] = [cur_pose.copy()]
    path: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    skipped: list[int] = []

    if not images_gray:
        return VOResult(poses, np.zeros((0, 3)), skipped)

    kp_prev, des_prev = detect_orb_grid(images_gray[0], n_grid, n_features_total)

    for i in range(1, len(images_gray)):
        kp_cur, des_cur = detect_orb_grid(images_gray[i], n_grid, n_features_total)
        good = match_descriptors(flann, des_prev, des_cur, lowe_ratio=lowe_ratio, min_matches=min_matches)

        ok = False
        if good:
            q1, q2 = matched_points(kp_prev, kp_cur, good)
            E, mask = essential_ransac(q1, q2, K, prob=essential_prob, thresh=essential_thresh)
            if E is not None and mask is not None and int(mask.sum()) >= 8:
                _, R, t, _ = cv2.recoverPose(E, q1, q2, K)
                if use_imu_scales and scales is not None and i < len(scales):
                    t = t * float(scales[i])
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t.ravel()
                cur_pose = cur_pose @ T
                ok = True
        if not ok:
            skipped.append(i)
            if verbose:
                print(f"[VO] frame {i}: pose non recuperata, mantengo posa precedente")

        poses.append(cur_pose.copy())
        path.append((float(cur_pose[0, 3]), float(cur_pose[1, 3]), float(cur_pose[2, 3])))
        kp_prev, des_prev = kp_cur, des_cur

    return VOResult(poses=poses, path=np.asarray(path, dtype=np.float64), skipped=skipped)
