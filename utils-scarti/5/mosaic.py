"""Mosaico ortografico su canvas pre-allocato in coordinate UTM-locali.

Idea:

  1. Per ogni frame, dalla `WorldPose` e dalla quota stimo il GSD del
     frame (m/pixel sul terreno).
  2. Risoluzione del canvas = GSD medio dei frame (m/pixel del mosaico).
  3. Calcolo i 4 angoli mondo (E, N) di ogni immagine e prendo l'AABB:
     quello e' il rettangolo che il canvas deve coprire (con un piccolo
     margine).
  4. Pre-alloco il canvas (zeri) e incollo ogni frame via
     `cv2.warpPerspective` ROI-locale, sovrascrivendo i pixel non neri
     (strategia "ultimo vince").

Assunzioni:
  - Camera nadir o quasi (pitch ~ -90). Per scatti molto obliqui questa
    proiezione affine non e' piu' una buona approssimazione e servirebbe
    una vera ortorettificazione su DEM.
  - Terreno piatto a quota `ground_alt_m`. Senza DEM e' il meglio che
    possiamo fare; per voli su terreno accidentato il mosaico avra'
    errori di parallasse proporzionali al rilievo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .io_loader import CameraCalibration
from .pose_fusion import WorldPose


@dataclass
class MosaicCanvas:
    """Buffer del mosaico in coordinate UTM-locali.

    `bounds_world` e' (xmin, ymin, xmax, ymax) in metri (relativi
    all'origine del proiettore UTM, coerente con `UtmFrame.east_rel/north_rel`).
    Il pixel (0, 0) del canvas corrisponde all'angolo (xmin, ymax) del mondo
    (l'asse Y mondo cresce verso l'alto, l'asse riga del canvas verso il
    basso).
    """

    image: np.ndarray                                       # (H, W, 3) uint8
    res_m_per_px: float
    bounds_world: tuple[float, float, float, float]         # xmin, ymin, xmax, ymax

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])


# --------------------------------------------------------------------------- #
# GSD, angoli mondo, planning
# --------------------------------------------------------------------------- #

def _gsd_per_frame(pose: WorldPose, K: np.ndarray, ground_alt_m: float) -> float:
    """Ground sample distance del frame in m/pixel.

    GSD = HAG / fx, dove HAG e' l'altezza sopra il terreno e fx il focale
    in pixel della K post-undistort. Si assume scatto nadir.
    """
    hag = pose.alt_m - ground_alt_m
    if hag <= 0:
        raise ValueError(
            f"Frame {pose.index}: HAG = {hag:.2f} m (<=0). "
            f"Controlla `ground_alt_m` (alt frame = {pose.alt_m:.2f})."
        )
    fx = float(K[0, 0])
    return hag / fx


def _world_corners(pose: WorldPose, K: np.ndarray,
                   img_size_wh: tuple[int, int],
                   ground_alt_m: float) -> np.ndarray:
    """Quattro angoli (E, N) mondo del frame, in metri UTM-locali. Shape (4, 2).

    Convenzione yaw DJI: 0 = Nord, + = senso orario.
      image-right_world = ( cos yaw, -sin yaw )
      image-up_world    = ( sin yaw,  cos yaw )
    image y (riga) cresce verso il basso, quindi delta_v si proietta sul
    versore image-down = -image-up.
    """
    W, H = img_size_wh
    cx, cy = float(K[0, 2]), float(K[1, 2])
    gsd = _gsd_per_frame(pose, K, ground_alt_m)

    yaw = math.radians(pose.yaw_deg)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)

    # (u, v) dei 4 angoli; ordine: TL, TR, BR, BL
    corners_uv = np.array([
        [0, 0], [W, 0], [W, H], [0, H],
    ], dtype=np.float64)

    du = corners_uv[:, 0] - cx
    dv = corners_uv[:, 1] - cy

    dE = gsd * ( du * cy_ - dv * sy_)
    dN = gsd * (-du * sy_ - dv * cy_)

    out = np.empty((4, 2), dtype=np.float64)
    out[:, 0] = pose.east_m + dE
    out[:, 1] = pose.north_m + dN
    return out


def plan_canvas(poses: list[WorldPose],
                calib: CameraCalibration,
                img_size_wh: tuple[int, int],
                ground_alt_m: float | None = None,
                margin_m: float = 5.0,
                max_pixels: int = 80_000_000,
                res_m_per_px: float | None = None) -> MosaicCanvas:
    """Pre-alloca il canvas a partire dalle pose e dalla calibrazione.

    Parameters
    ----------
    poses : list[WorldPose]
        Pose fuse di tutti i frame da includere.
    calib : CameraCalibration
        Intrinseci post-undistort.
    img_size_wh : tuple[int, int]
        Dimensione (W, H) delle immagini undistorted. Deve essere
        uniforme su tutta la sequenza.
    ground_alt_m : float | None
        Quota del terreno in m s.l.m. (stesso riferimento di `alt_m` nelle
        FrameMeta). Se None, uso la quota del primo frame meno 1 m come
        proxy "drone appena decollato".
    margin_m : float
        Padding in metri attorno all'AABB dei frame.
    max_pixels : int
        Tetto di sicurezza sul totale pixel canvas: se superato, alzo
        ValueError con suggerimento di rivedere la GSD o di croppare la
        sequenza. Evita allocazioni accidentali da decine di GB.
    res_m_per_px : float | None
        Override esplicito della risoluzione del canvas (m/pixel). Se None
        uso il GSD medio dei frame (nitidezza ottimale, dimensioni variabili);
        se passato, lo applico tale e quale (utile per fissare un canvas
        piu' grossolano e leggero).
    """
    if not poses:
        raise ValueError("`poses` vuota")
    K = calib.K
    if ground_alt_m is None:
        # Convenzione minima: il drone non scatta nel terreno; assumo
        # un metro sotto al primo scatto come "ground reference".
        ground_alt_m = poses[0].alt_m - 1.0

    if res_m_per_px is not None:
        if res_m_per_px <= 0:
            raise ValueError(f"res_m_per_px deve essere > 0, ricevuto {res_m_per_px}")
        res = float(res_m_per_px)
    else:
        # GSD canvas = media dei GSD per frame.
        gsd_list = np.array(
            [_gsd_per_frame(p, K, ground_alt_m) for p in poses],
            dtype=np.float64,
        )
        res = float(gsd_list.mean())

    # AABB mondo di tutti i frame.
    all_corners = np.vstack([
        _world_corners(p, K, img_size_wh, ground_alt_m) for p in poses
    ])
    xmin = float(all_corners[:, 0].min()) - margin_m
    xmax = float(all_corners[:, 0].max()) + margin_m
    ymin = float(all_corners[:, 1].min()) - margin_m
    ymax = float(all_corners[:, 1].max()) + margin_m

    W_canvas = int(math.ceil((xmax - xmin) / res))
    H_canvas = int(math.ceil((ymax - ymin) / res))
    if W_canvas * H_canvas > max_pixels:
        gb = W_canvas * H_canvas * 3 / 1e9
        raise ValueError(
            f"Canvas {W_canvas}x{H_canvas} px ({gb:.1f} GB BGR) supera "
            f"max_pixels={max_pixels}. Aumenta `margin_m`, riduci il range "
            "di frame o usa una risoluzione fissa piu' grossolana."
        )

    image = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
    return MosaicCanvas(image=image, res_m_per_px=res,
                        bounds_world=(xmin, ymin, xmax, ymax))


# --------------------------------------------------------------------------- #
# Warp di un singolo frame sul canvas
# --------------------------------------------------------------------------- #

def _homography_image_to_canvas(pose: WorldPose,
                                K: np.ndarray,
                                canvas: MosaicCanvas,
                                ground_alt_m: float) -> np.ndarray:
    """3x3 che mappa pixel immagine -> pixel canvas (affine, embedded come H).

    Derivazione: vedi `_world_corners` per la composizione tra delta-pixel
    dell'immagine e versori (image-right, image-down) in coordinate mondo;
    qui aggiungiamo la conversione mondo -> pixel canvas (Y flip).
    """
    cx, cy = float(K[0, 2]), float(K[1, 2])
    gsd = _gsd_per_frame(pose, K, ground_alt_m)
    s = gsd / canvas.res_m_per_px

    yaw = math.radians(pose.yaw_deg)
    cy_, sy_ = math.cos(yaw), math.sin(yaw)

    # col = a*u + b*v + (e - a*cx - b*cy)
    # row = c*u + d*v + (f - c*cx - d*cy)
    a = s * cy_
    b = -s * sy_
    c = s * sy_
    d = s * cy_

    xmin, _ymin, _xmax, ymax = canvas.bounds_world
    e_const = (pose.east_m - xmin) / canvas.res_m_per_px
    f_const = (ymax - pose.north_m) / canvas.res_m_per_px

    e = e_const - a * cx - b * cy
    f = f_const - c * cx - d * cy

    return np.array([
        [a, b, e],
        [c, d, f],
        [0, 0, 1],
    ], dtype=np.float64)


def paste_frame(canvas: MosaicCanvas,
                image_bgr: np.ndarray,
                pose: WorldPose,
                calib: CameraCalibration,
                ground_alt_m: float) -> None:
    """Warpa `image_bgr` sul canvas e sovrascrive i pixel non neri.

    Strategia "ultimo vince": ogni nuovo frame copre i pixel precedenti.
    Per evitare di allocare un buffer della dimensione del canvas ogni
    volta, calcolo il bbox della proiezione e warpo solo nella ROI.
    """
    H_img, W_img = image_bgr.shape[:2]
    H_full = _homography_image_to_canvas(pose, calib.K, canvas, ground_alt_m)

    # Proietto i 4 angoli per stimare la ROI sul canvas.
    corners = np.array([
        [0, 0, 1], [W_img, 0, 1], [W_img, H_img, 1], [0, H_img, 1],
    ], dtype=np.float64).T  # 3x4
    proj = H_full @ corners
    cols, rows = proj[0], proj[1]

    col_min = max(0, int(math.floor(cols.min())))
    col_max = min(canvas.width, int(math.ceil(cols.max())))
    row_min = max(0, int(math.floor(rows.min())))
    row_max = min(canvas.height, int(math.ceil(rows.max())))
    if col_max <= col_min or row_max <= row_min:
        # Il frame cade interamente fuori canvas (margini stretti).
        return

    # Traslo la homography nello spazio della ROI: H_roi = T(-col_min,-row_min) @ H_full
    T = np.array([
        [1, 0, -col_min],
        [0, 1, -row_min],
        [0, 0, 1],
    ], dtype=np.float64)
    H_roi = T @ H_full

    roi_w = col_max - col_min
    roi_h = row_max - row_min
    warped = cv2.warpPerspective(
        image_bgr, H_roi, (roi_w, roi_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    # Maschera dei pixel davvero coperti dal frame (gli zero sono "vuoto").
    mask = warped.any(axis=2)
    if not mask.any():
        return

    canvas.image[row_min:row_max, col_min:col_max][mask] = warped[mask]


def crop_to_content(canvas: MosaicCanvas) -> MosaicCanvas:
    """Ritaglia il canvas al bbox dei pixel non neri.

    Aggiorna `bounds_world` di conseguenza, cosi' il GeoTIFF prodotto da
    `geotiff_writer.scrivi_geotiff` resta correttamente georeferenziato.
    Convenzione: il pixel (0, 0) del canvas corrisponde al corner
    (xmin, ymax) del mondo; la riga cresce verso il basso, la Y mondo
    verso l'alto.

    Modifica `canvas` in place e lo restituisce per comodita' di chaining.
    """
    mask = canvas.image.any(axis=2)
    if not mask.any():
        return canvas  # canvas vuoto, nulla da ritagliare

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1

    res = canvas.res_m_per_px
    xmin, _ymin_old, _xmax_old, ymax = canvas.bounds_world
    new_xmin = xmin + c0 * res
    new_xmax = xmin + c1 * res
    new_ymax = ymax - r0 * res
    new_ymin = ymax - r1 * res

    canvas.image = canvas.image[r0:r1, c0:c1].copy()
    canvas.bounds_world = (new_xmin, new_ymin, new_xmax, new_ymax)
    return canvas


def build_mosaic(poses: list[WorldPose],
                 images_bgr_iter,
                 calib: CameraCalibration,
                 img_size_wh: tuple[int, int],
                 ground_alt_m: float | None = None,
                 margin_m: float = 5.0,
                 skip_low_confidence: bool = False,
                 max_pixels: int = 80_000_000,
                 res_m_per_px: float | None = None) -> MosaicCanvas:
    """Costruisce il mosaico end-to-end: planning + paste in sequenza.

    Parameters
    ----------
    images_bgr_iter
        Iterabile che yield (pose_index, image_bgr) ALLINEATO a `poses`.
        Tipicamente costruito sopra `FrameStream`.
    skip_low_confidence
        Se True, ignora i frame con `pose.confident == False` (utile per
        sanity check dell'IMU/VO; di default sovrascriviamo tutto).
    max_pixels, res_m_per_px
        Inoltrati a `plan_canvas`.
    """
    canvas = plan_canvas(poses, calib, img_size_wh,
                         ground_alt_m=ground_alt_m, margin_m=margin_m,
                         max_pixels=max_pixels, res_m_per_px=res_m_per_px)
    if ground_alt_m is None:
        ground_alt_m = poses[0].alt_m - 1.0  # stesso default di plan_canvas

    pose_by_index = {p.index: p for p in poses}
    for pose_index, img in images_bgr_iter:
        pose = pose_by_index.get(pose_index)
        if pose is None:
            continue
        if skip_low_confidence and not pose.confident:
            continue
        paste_frame(canvas, img, pose, calib, ground_alt_m)

    return canvas
