"""Raffinamento globale delle pose via least-squares.

Ogni frame e' parametrizzato da (tx, ty, theta, log_scale). Si minimizza la somma di:
- residuo "GPS anchor": angoli del frame nella posa attuale vs angoli nella posa iniziale GPS,
  pesato da gps_weight (controlla quanto fidarsi del GPS).
- residuo "vincolo pairwise": per ogni coppia di vicini con similarity H_ij dalle feature,
  M_i @ corners_i deve coincidere con M_j @ H_ij @ corners_i, pesato per il sqrt del numero
  di inlier (piu' inlier = vincolo piu' affidabile).

Il problema e' sparso: i parametri di un frame influenzano solo i suoi residui anchor e quelli
delle coppie in cui appare. Usiamo jac_sparsity per accelerare scipy.
"""
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def _decompose_similarity_3x3(M: np.ndarray) -> tuple[float, float, float, float]:
    a, b = float(M[0, 0]), float(M[1, 0])
    scale = math.hypot(a, b)
    return float(M[0, 2]), float(M[1, 2]), math.atan2(b, a), math.log(max(scale, 1e-12))


def _compose_similarity_3x3(tx: float, ty: float, theta: float, log_scale: float) -> np.ndarray:
    s = math.exp(log_scale)
    c, sn = math.cos(theta), math.sin(theta)
    return np.array(
        [[s * c, -s * sn, tx], [s * sn, s * c, ty], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _build_M_array(params: np.ndarray, n: int) -> np.ndarray:
    """params shape (n*4,) -> array (n, 3, 3) di matrici similarity."""
    p = params.reshape(n, 4)
    tx, ty, theta, log_s = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    s = np.exp(log_s)
    c, sn = np.cos(theta), np.sin(theta)
    Ms = np.zeros((n, 3, 3), dtype=np.float64)
    Ms[:, 0, 0] = s * c
    Ms[:, 0, 1] = -s * sn
    Ms[:, 1, 0] = s * sn
    Ms[:, 1, 1] = s * c
    Ms[:, 0, 2] = tx
    Ms[:, 1, 2] = ty
    Ms[:, 2, 2] = 1.0
    return Ms


def _apply_M_batch(Ms: np.ndarray, pts_h: np.ndarray) -> np.ndarray:
    """Ms: (n, 3, 3), pts_h: (K, 3). Out: (n, K, 2) — applica ogni M agli stessi punti."""
    # einsum: n3a, K3 -> nKa  (su indice 3 della M e ultimo di pts)
    out = np.einsum("nab,kb->nka", Ms, pts_h)
    return out[..., :2]


def refine_poses(
    initial_M: list[np.ndarray],
    pair_constraints: list[tuple[int, int, np.ndarray, int]],
    image_size: tuple[int, int],
    gps_weight: float = 1.0,
    max_iter: int = 30,
    verbose: int = 1,
) -> list[np.ndarray]:
    """Ritorna pose raffinate.

    pair_constraints: lista di (i, j, H_ij, n_inliers) dove H_ij mappa pixel_image_i ->
    pixel_image_j.
    """
    n = len(initial_M)
    if n == 0:
        return []
    w, h = image_size

    corners = np.array(
        [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]], dtype=np.float64
    )  # (4, 3) homogenei

    # Parametri iniziali
    x0 = np.zeros(n * 4, dtype=np.float64)
    for i, M in enumerate(initial_M):
        x0[i * 4: i * 4 + 4] = _decompose_similarity_3x3(M)

    # Target dei residui anchor: dove finiscono i corner con la posa iniziale
    initial_Ms = np.array(initial_M, dtype=np.float64)  # (n, 3, 3)
    gps_corners = _apply_M_batch(initial_Ms, corners)  # (n, 4, 2)

    # Pre-applica H_ij ai corner per ogni vincolo
    constraints = []
    for (i, j, H_ij, n_inl) in pair_constraints:
        corners_in_j = (corners @ H_ij.T)[:, :2]  # (4, 2)
        corners_in_j_h = np.column_stack([corners_in_j, np.ones(4)])  # (4, 3)
        weight = float(np.sqrt(max(n_inl, 1)))
        constraints.append((i, j, corners_in_j_h, weight))

    n_res_anchor = n * 8  # 4 corner x 2 coord per ogni frame
    n_res_pairs = len(constraints) * 8
    n_residuals = n_res_anchor + n_res_pairs
    n_params = n * 4

    def residuals(params: np.ndarray) -> np.ndarray:
        Ms = _build_M_array(params, n)
        # Anchor residuals
        curr_corners = _apply_M_batch(Ms, corners)  # (n, 4, 2)
        anchor_res = (curr_corners - gps_corners) * gps_weight  # (n, 4, 2)
        out = [anchor_res.reshape(-1)]
        # Pair residuals
        if constraints:
            pair_res = np.zeros((len(constraints), 4, 2), dtype=np.float64)
            for k, (i, j, c_in_j_h, w_p) in enumerate(constraints):
                pts_i = corners @ Ms[i].T  # (4, 3)
                pts_j = c_in_j_h @ Ms[j].T  # (4, 3)
                pair_res[k] = (pts_i[:, :2] - pts_j[:, :2]) * w_p
            out.append(pair_res.reshape(-1))
        return np.concatenate(out)

    # Sparsita' del jacobiano: ogni residuo dipende solo dai param dei frame coinvolti
    jac = lil_matrix((n_residuals, n_params), dtype=np.uint8)
    for i in range(n):
        jac[i * 8: (i + 1) * 8, i * 4: (i + 1) * 4] = 1
    for k, (i, j, _, _) in enumerate(constraints):
        row0 = n_res_anchor + k * 8
        jac[row0: row0 + 8, i * 4: (i + 1) * 4] = 1
        jac[row0: row0 + 8, j * 4: (j + 1) * 4] = 1
    jac_csr = jac.tocsr()

    result = least_squares(
        residuals,
        x0,
        jac_sparsity=jac_csr,
        method="trf",
        max_nfev=max_iter,
        verbose=verbose,
        xtol=1e-7,
        ftol=1e-7,
    )

    refined = []
    for i in range(n):
        tx, ty, theta, log_s = result.x[i * 4: (i + 1) * 4]
        refined.append(_compose_similarity_3x3(tx, ty, theta, log_s))
    return refined
