"""Costruzione di un chunk-panorama per una singola passata.

Scelta implementativa per V1: **rettifica diretta per-frame** su una griglia
UTM-aligned (modello flat-earth, gimbal nadir). Per imagery nadir UAV a quota
sostanzialmente costante questa scelta e' equivalente in qualita' a un
cv2.Stitcher su singola passata, ma:

    - e' deterministica (non si "rifiuta" senza spiegazione come Stitcher)
    - produce chunk gia' in coordinate metriche (UTM), quindi la composizione
      globale diventa una copia di pixel
    - non richiede di estrarre i parametri interni del warper di Stitcher per
      conoscere la posizione di ogni frame nel panorama

Lascio uno stub commentato per integrare cv2.Stitcher come motore alternativo
in una V2 futura: in tal caso servira' un passaggio aggiuntivo di matching
frame->panorama per agganciare il chunk al sistema UTM tramite Umeyama sui
centri dei frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Chunk:
    """Chunk = porzione di mosaico relativa a una passata.

    Il chunk vive in un piano UTM locale: un pixel del chunk corrisponde a
    `gsd_m_per_px` metri sul terreno. L'origine (pixel 0, 0) corrisponde
    all'angolo `(min_E, max_N)` in UTM.
    """
    image: np.ndarray            # BGR uint8
    coverage: np.ndarray         # mappa di copertura uint8 0..255 (255 = pieno)
    min_E: float                 # UTM Est dell'angolo top-left (m)
    max_N: float                 # UTM Nord dell'angolo top-left (m)
    gsd_m_per_px: float
    pass_index: int
    frame_indices: list[int]


def _frame_to_utm_affine(
    K: np.ndarray,
    yaw_deg: float,
    gps_utm_xy: tuple[float, float],
    agl_m: float,
    yaw_sign: int = 1,
) -> np.ndarray:
    """Affine 2x3 che mappa pixel del frame -> coordinate UTM (E, N), in metri.

    Assunzioni:
      * gimbal nadir (camera che guarda dritta verso il basso)
      * terreno piatto a distanza AGL costante
      * yaw_deg = heading drone in gradi (DJI: 0 = Nord, +CW)

    `yaw_sign` controlla la direzione effettiva di rotazione del frame nel
    piano (in caso di convenzione opposta tra IMU/gimbal): +1 di default.
    """
    f = float(K[0, 0])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    s = agl_m / f          # GSD del frame: metri di terreno per pixel immagine

    h = np.radians(yaw_sign * yaw_deg)
    cos_h = np.cos(h)
    sin_h = np.sin(h)

    E0, N0 = gps_utm_xy

    # Direzione "destra del drone" in UTM (E, N) per heading h CW da Nord:
    #   r_world = (cos h, -sin h)   <-- legata a image_x
    # Direzione "indietro del drone" (camera y nadir tipica DJI) in UTM:
    #   b_world = (-sin h, -cos h)  <-- legata a image_y
    # E = E0 + s * (cos_h * (x - cx) - sin_h * (y - cy))
    # N = N0 + s * (-sin_h * (x - cx) - cos_h * (y - cy))
    A = np.array([
        [s * cos_h,  -s * sin_h,  E0 - s * cos_h * cx + s * sin_h * cy],
        [-s * sin_h, -s * cos_h,  N0 + s * sin_h * cx + s * cos_h * cy],
    ], dtype=np.float64)
    return A


def _utm_aabb_for_frames(
    affines: list[np.ndarray],
    img_shape_hw: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Bounding box (min_E, min_N, max_E, max_N) di tutti i 4 angoli di tutti i frame in UTM."""
    h, w = img_shape_hw
    corners_px = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    all_pts = []
    for A in affines:
        pts = (A @ corners_px).T  # (4, 2)
        all_pts.append(pts)
    P = np.vstack(all_pts)
    return float(P[:, 0].min()), float(P[:, 1].min()), float(P[:, 0].max()), float(P[:, 1].max())


def build_chunk_for_pass(
    frame_paths: list[Path | str],
    yaws_deg: list[float],
    gps_utm: np.ndarray,                # (N, 2) [E, N] in metri
    agl_m: float,
    K: np.ndarray,
    *,
    pass_index: int,
    frame_indices: list[int],
    canvas_gsd_m_per_px: float = 0.05,
    yaw_sign: int = 1,
    save_path: Path | str | None = None,
) -> Chunk | None:
    """Crea un chunk per una passata composendo i frame su griglia UTM-aligned.

    Per ogni frame calcola l'affine pixel->UTM, costruisce l'affine inverso
    per warpAffine in canvas-pixel, e fa average-blending nel canvas.
    """
    if not frame_paths:
        return None

    # Calcola affine per ogni frame
    affines_utm: list[np.ndarray] = []
    img_shape_hw: tuple[int, int] | None = None
    for i, p in enumerate(frame_paths):
        # Apriamo solo il primo per leggere shape; gli altri si assumono uguali
        if img_shape_hw is None:
            img0 = cv2.imread(str(p))
            if img0 is None:
                raise FileNotFoundError(f"Impossibile leggere {p}")
            img_shape_hw = (img0.shape[0], img0.shape[1])
        A = _frame_to_utm_affine(
            K, float(yaws_deg[i]), (float(gps_utm[i, 0]), float(gps_utm[i, 1])),
            agl_m, yaw_sign=yaw_sign,
        )
        affines_utm.append(A)

    assert img_shape_hw is not None
    min_E, min_N, max_E, max_N = _utm_aabb_for_frames(affines_utm, img_shape_hw)

    # Canvas
    g = float(canvas_gsd_m_per_px)
    W = int(np.ceil((max_E - min_E) / g)) + 2
    H = int(np.ceil((max_N - min_N) / g)) + 2
    if W <= 0 or H <= 0 or W * H > 250_000_000:
        # Sanita': scarta chunk patologici (>250 Mpix)
        return None
    canvas = np.zeros((H, W, 3), dtype=np.uint16)
    counts = np.zeros((H, W), dtype=np.uint16)

    # Trasformazione UTM -> pixel canvas: pixel = ((E - min_E)/g, (max_N - N)/g)
    # Quindi affine pixel_frame -> pixel_canvas = M_utm2px @ A
    M_utm2px = np.array([[1.0 / g, 0.0, -min_E / g],
                         [0.0, -1.0 / g, max_N / g]], dtype=np.float64)

    for i, p in enumerate(frame_paths):
        img = cv2.imread(str(p))
        if img is None:
            continue
        A_utm = affines_utm[i]
        # Componi: pixel_frame -> pixel_canvas
        # In forma 3x3 per la composizione:
        A3 = np.vstack([A_utm, [0, 0, 1]])
        Mt = np.vstack([M_utm2px, [0, 0, 1]])
        Acomp = (Mt @ A3)[:2, :]
        warped = cv2.warpAffine(
            img, Acomp, (W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # Maschera dei pixel validi (non zero)
        mask = (warped.sum(axis=2) > 0).astype(np.uint16)
        canvas += warped.astype(np.uint16) * mask[..., None]
        counts += mask

    counts_safe = np.maximum(counts, 1).astype(np.uint16)
    out = (canvas // counts_safe[..., None]).astype(np.uint8)
    coverage = np.minimum(counts.astype(np.uint16) * 64, 255).astype(np.uint8)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out)

    return Chunk(
        image=out,
        coverage=coverage,
        min_E=float(min_E),
        max_N=float(max_N),
        gsd_m_per_px=float(g),
        pass_index=pass_index,
        frame_indices=list(frame_indices),
    )
