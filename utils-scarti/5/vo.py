"""Visual odometry frame-by-frame, scalato con i passi metrici da GPS/UTM.

Per ogni coppia di frame consecutivi, dai match inlier ricaviamo (R, t)
con `findEssentialMat` + `recoverPose`. La traslazione di `recoverPose` ha
modulo unitario: la scaliamo a |Delta UTM| consecutivo perche' il mosaico
finale viva in metri.

Il modulo NON sa nulla di UTM o di pose globali: ritorna solo lo *step*
relativo nel frame della camera precedente. La conversione a coordinate
mondo (rotazione di t verso est/nord, agganciamento al GPS) e' compito di
`pose_fusion`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .matching import MatchResult


@dataclass(frozen=True)
class VoStep:
    """Moto relativo tra il frame i-1 e il frame i.

    Convenzione `cv2.recoverPose`: dati i match (q1, q2), la posa stimata
    e' quella della camera 2 espressa nel riferimento della camera 1. In
    altre parole, un punto X1 nel frame 1 corrisponde a X2 = R @ X1 + t
    nel frame 2. Il vettore `t` qui e' gia' scalato in metri.
    """

    R: np.ndarray                 # 3x3 float64, rotazione cam2 wrt cam1
    t: np.ndarray                 # (3,) float64, traslazione in metri
    n_inliers: int                # inlier dopo il chirality check di recoverPose
    ok: bool                      # False se la stima non e' affidabile


def transf(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Matrice omogenea 4x4 a partire da R (3x3) e t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=np.float64).ravel()
    return T


def estimate_step(match: MatchResult,
                  K: np.ndarray,
                  scale_m: float,
                  ransac_prob: float = 0.999,
                  ransac_thresh_px: float = 1.0,
                  min_inliers: int = 8) -> VoStep:
    """Stima (R, t_metri) da un `MatchResult` e dalla scala metrica GPS.

    Parameters
    ----------
    match : MatchResult
        Match inlier filtrati gia' da `matching.match_frames`.
    K : np.ndarray
        Camera matrix 3x3 *post-undistort* (quella di `CameraCalibration.K`).
    scale_m : float
        Distanza metrica reale percorsa tra il frame i-1 e il frame i
        (da `passi_metrici(utm_frames)[i-1]`). Se ~0 (drone fermo) la
        traslazione viene mantenuta nulla e `ok` resta True: la rotazione
        e' comunque valida.
    ransac_prob, ransac_thresh_px : float
        Parametri di `cv2.findEssentialMat` (RANSAC).
    min_inliers : int
        Soglia sotto la quale `ok=False`: con pochi inlier la decomposizione
        chirality di `recoverPose` e' rumorosa.
    """
    if K.shape != (3, 3):
        raise ValueError(f"K deve essere 3x3, ricevuto {K.shape}")
    if scale_m < 0:
        raise ValueError(f"scale_m deve essere >= 0, ricevuto {scale_m}")

    if len(match) < min_inliers:
        return VoStep(R=np.eye(3), t=np.zeros(3), n_inliers=len(match), ok=False)

    E, mask_e = cv2.findEssentialMat(
        match.pts1, match.pts2, K,
        method=cv2.RANSAC, prob=ransac_prob, threshold=ransac_thresh_px,
    )
    if E is None or mask_e is None:
        return VoStep(np.eye(3), np.zeros(3), 0, ok=False)

    # `recoverPose` ri-filtra in base al chirality test (Z > 0 in entrambe
    # le camere). Passargli la stessa mask_e e' raccomandato in OpenCV >=4.
    n_inl, R, t_unit, _ = cv2.recoverPose(
        E, match.pts1, match.pts2, K, mask=mask_e,
    )
    if n_inl < min_inliers:
        return VoStep(R=np.eye(3), t=np.zeros(3), n_inliers=int(n_inl), ok=False)

    # `t_unit` ha modulo 1; lo scaliamo alla distanza UTM. La direzione
    # resta espressa nel sistema della camera precedente: la rotazione
    # verso est/nord avviene in pose_fusion.
    t = (t_unit.ravel() * float(scale_m)).astype(np.float64)

    return VoStep(R=R.astype(np.float64), t=t, n_inliers=int(n_inl), ok=True)


def accumulate(steps: list[VoStep],
               initial_pose: np.ndarray | None = None) -> list[np.ndarray]:
    """Compone gli step in pose cumulate 4x4 a partire da `initial_pose`.

    `pose[i] = pose[i-1] @ transf(R_i, t_i)`. Il sistema di riferimento
    e' quello della camera al frame 0 (NON UTM). Step con `ok=False`
    vengono saltati replicando la posa precedente: il caller li sostituira'
    tipicamente con la posizione GPS in `pose_fusion`.
    """
    pose = np.eye(4, dtype=np.float64) if initial_pose is None else initial_pose.copy()
    out = [pose.copy()]
    for s in steps:
        if s.ok:
            pose = pose @ transf(s.R, s.t)
        out.append(pose.copy())
    return out
