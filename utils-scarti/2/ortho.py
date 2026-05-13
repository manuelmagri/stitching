"""Ortomosaico geo-riferito da GPS + gimbal yaw + altitudine AGL.

Assume gimbal pitch ~ -90° (nadir): la trasformazione immagine→terreno è una
**similarità** (rotazione + scala uniforme + traslazione), non un'omografia.
Per ogni frame:

    pixel_immagine --[centra a (cx, cy)]--> coord camera in pixel
                   --[rotazione di yaw]----> orientate al nord
                   --[scala m/px = h/f]----> coord ENU relative al centro frame
                   --[traslazione GPS]-----> coord ENU mondo
                   --[canvas (m/px = gsd)]-> pixel canvas

L'altitudine AGL (h) viene auto-calibrata da `auto_calibrate_agl` confrontando
spostamenti GPS metrici e spostamenti pixel via feature matching.
"""
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .io_calibration import FrameGeometry
from .timing import timed


@dataclass
class OrthoResult:
    """Output dell'ortomosaico, con i parametri del canvas per geo-referenziazione."""
    image: np.ndarray         # (H, W, 3) uint8
    gsd: float                # ground sampling distance, metri/pixel
    east_min: float           # est ENU del pixel (0, 0) del canvas
    north_max: float          # nord ENU del pixel (0, 0) — y cresce verso sud
    agl: float                # altitudine sul terreno usata, in metri


# -------------------------------------------------------------------- #
# Auto-calibrazione AGL                                                #
# -------------------------------------------------------------------- #

@timed
def auto_calibrate_agl(
    images: List[np.ndarray],
    geometry: FrameGeometry,
    focal_px: float,
    *,
    yaw_stability_deg: float = 1.0,
    min_displacement_m: float = 1.0,
    min_inliers: int = 20,
    max_pairs: int = 30,
) -> float:
    """Stima l'altitudine sul terreno (AGL) confrontando spostamenti GPS e pixel.

    Itera su coppie consecutive di frame "in crociera" (yaw stabile, drone in
    movimento). Per ognuna stima la traslazione pixel via `estimateAffinePartial2D`
    su match ORB. Poi calcola:

        m_per_pixel = ‖ΔGPS‖ / ‖Δpx‖
        h_AGL       = m_per_pixel · focale

    Aggrega con la mediana per robustezza agli outlier (singoli match cattivi).

    Args:
        images: frame BGR allineati a `geometry`.
        geometry: posizioni ENU + yaw per ogni frame.
        focal_px: focale in pixel (cameraMatrix[0, 0]) **alla risoluzione delle
            immagini in `images`**. Se le immagini sono ridimensionate di un
            fattore `r`, questo deve essere `K[0,0] / r`.
        yaw_stability_deg: scarta coppie con |Δyaw| > soglia (filtra le virate).
        min_displacement_m: scarta coppie con drone quasi fermo.
        min_inliers: minimo di inliers RANSAC per ritenere la stima affidabile.
        max_pairs: ferma la raccolta dopo questo numero di coppie valide.

    Solleva `RuntimeError` se nessuna coppia produce una stima valida.
    """
    if len(images) != len(geometry.positions_enu):
        raise ValueError("images e geometry devono avere la stessa lunghezza.")
    if len(images) < 2:
        raise ValueError("Servono almeno 2 immagini per la calibrazione AGL.")

    orb = cv2.ORB_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ratios = []
    for i in range(1, len(images)):
        if len(ratios) >= max_pairs:
            break

        dyaw = geometry.yaw_deg[i] - geometry.yaw_deg[i - 1]
        # Wrap in [-180, 180] per gestire il salto attorno a ±180°.
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        if abs(dyaw) > yaw_stability_deg:
            continue

        dE = geometry.positions_enu[i, 0] - geometry.positions_enu[i - 1, 0]
        dN = geometry.positions_enu[i, 1] - geometry.positions_enu[i - 1, 1]
        gps_disp_m = float(np.hypot(dE, dN))
        if gps_disp_m < min_displacement_m:
            continue

        gray_a = cv2.cvtColor(images[i - 1], cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)
        if des_a is None or des_b is None or len(des_a) < min_inliers or len(des_b) < min_inliers:
            continue

        matches = matcher.match(des_a, des_b)
        if len(matches) < min_inliers:
            continue
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in matches])

        M, inliers = cv2.estimateAffinePartial2D(
            pts_a, pts_b, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None or inliers is None or int(inliers.sum()) < min_inliers:
            continue

        # Shift puro mediano sui soli inlier RANSAC: indipendente dalla
        # piccola rotazione/scala residua di M (yaw_stability_deg < 1° ma
        # comunque non zero), che altrimenti contamina (tx, ty).
        inl = inliers.ravel().astype(bool)
        shift = np.median(pts_b[inl] - pts_a[inl], axis=0)
        pixel_disp = float(np.hypot(shift[0], shift[1]))
        if pixel_disp < 1.0:
            continue

        ratios.append(gps_disp_m / pixel_disp)

    if not ratios:
        raise RuntimeError(
            "Auto-calibrazione AGL fallita: nessuna coppia valida. "
            "Controlla yaw_stability_deg / min_displacement_m / numero immagini."
        )

    m_per_pixel = float(np.median(ratios))
    agl = m_per_pixel * focal_px
    print(
        f"[ortho] AGL auto-calibrata: {agl:.1f} m "
        f"(m/px = {m_per_pixel:.4f}, da {len(ratios)} coppie, "
        f"std relativa = {np.std(ratios) / m_per_pixel:.2%})"
    )
    return agl


# -------------------------------------------------------------------- #
# Composizione ortomosaico                                             #
# -------------------------------------------------------------------- #

def _pixel_to_enu_coeffs(K_scaled, yaw_deg, position_en, agl):
    """Coefficienti dell'affine pixel-immagine → ENU per un singolo frame.

    Convenzione (DJI / aviazione, gimbal in FollowYaw):
      - yaw_deg = compass bearing della direzione "image-top", +cw da nord
        (corrisponde a FlightYawDegree quando il gimbal segue il body).
      - Image axes: u = right (x_image), v = down (y_image).
      - Nadir gimbal (pitch -90°): la proiezione a terra è una similarità.

    Vettori-base in mondo ENU (per yaw=0: image-top punta a nord):
      image-right  → R_cw(yaw) · (+E)  = ( cos yaw, -sin yaw)
      image-down   → R_cw(yaw) · (-N)  = (-sin yaw, -cos yaw)

    Pixel (u, v) → offset metrico:
      ΔE = s·(u-cx)·cos yaw  -  s·(v-cy)·sin yaw
      ΔN = -s·(u-cx)·sin yaw -  s·(v-cy)·cos yaw      (s = AGL / f)

    Ritorna (a11, a12, a21, a22, b1, b2) tali che:
      E = a11·u + a12·v + b1
      N = a21·u + a22·v + b2
    """
    cx, cy = K_scaled[0, 2], K_scaled[1, 2]
    f = K_scaled[0, 0]
    s = agl / f
    c = np.cos(np.radians(yaw_deg))
    sn = np.sin(np.radians(yaw_deg))

    a11 = s * c
    a12 = -s * sn
    a21 = -s * sn
    a22 = -s * c
    b1 = position_en[0] - cx * a11 - cy * a12
    b2 = position_en[1] - cx * a21 - cy * a22
    return a11, a12, a21, a22, b1, b2


def _frame_corners_world(image_shape, K_scaled, yaw_deg, position_en, agl):
    """Coordinate ENU dei 4 angoli del frame proiettato a terra."""
    h, w = image_shape[:2]
    a11, a12, a21, a22, b1, b2 = _pixel_to_enu_coeffs(K_scaled, yaw_deg, position_en, agl)
    corners_uv = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    u, v = corners_uv[:, 0], corners_uv[:, 1]
    E = a11 * u + a12 * v + b1
    N = a21 * u + a22 * v + b2
    return np.stack([E, N], axis=1)


def _affine_image_to_canvas(K_scaled, yaw_deg, position_en, agl, gsd, east_min, north_max):
    """2x3 affine pixel sorgente → pixel canvas.

    Composizione: (E, N) → (X_canvas, Y_canvas)
      X_canvas = (E - east_min) / gsd
      Y_canvas = (north_max - N) / gsd     (Y canvas cresce verso sud)
    """
    a11, a12, a21, a22, b1, b2 = _pixel_to_enu_coeffs(K_scaled, yaw_deg, position_en, agl)
    return np.array([
        [ a11 / gsd,  a12 / gsd, (b1 - east_min) / gsd],
        [-a21 / gsd, -a22 / gsd, (north_max - b2) / gsd],
    ], dtype=np.float64)


@timed
def render_ortho(
    images: List[np.ndarray],
    geometry: FrameGeometry,
    K_scaled: np.ndarray,
    *,
    agl: Optional[float] = None,
    gsd: Optional[float] = None,
    auto_calib_kwargs: Optional[dict] = None,
) -> OrthoResult:
    """Compone l'ortomosaico geo-riferito con blending best-pixel (Voronoi).

    Per ogni pixel del canvas, sceglie il frame il cui contributo è più "centrale"
    (massima distanza dai bordi del frame). Niente media, niente fantasma da
    sovrapposizioni. Le cuciture cadono lungo le bisettrici dei centri dei frame.

    Args:
        images: frame BGR ordinati.
        geometry: posizioni ENU e yaw allineati a `images`.
        K_scaled: matrice intrinseca *alla risoluzione delle immagini in input*.
            Se hai ridimensionato con factor `r`, passa K con fx/fy/cx/cy /= r.
        agl: altitudine sul terreno in metri. Se None, viene auto-calibrata.
        gsd: ground sampling distance del canvas (m/pixel). Se None, default a
            (AGL / focale): la stessa scala dei frame originali (no upscaling).
        auto_calib_kwargs: argomenti extra per `auto_calibrate_agl`.
    """
    if len(images) != len(geometry.positions_enu):
        raise ValueError("images e geometry devono avere la stessa lunghezza.")
    if len(images) < 1:
        raise ValueError("Almeno un frame richiesto.")

    focal_px = float(K_scaled[0, 0])
    if agl is None:
        agl = auto_calibrate_agl(images, geometry, focal_px, **(auto_calib_kwargs or {}))
    if gsd is None:
        gsd = agl / focal_px  # nessun upscaling rispetto al frame originale

    # Bounding box mondo in ENU dalle pose geometriche.
    all_corners = []
    for i, img in enumerate(images):
        corners = _frame_corners_world(
            img.shape, K_scaled,
            float(geometry.yaw_deg[i]),
            geometry.positions_enu[i],
            agl,
        )
        all_corners.append(corners)
    all_corners = np.concatenate(all_corners, axis=0)
    east_min, east_max = all_corners[:, 0].min(), all_corners[:, 0].max()
    north_min, north_max = all_corners[:, 1].min(), all_corners[:, 1].max()

    M_use = [
        _affine_image_to_canvas(
            K_scaled,
            float(geometry.yaw_deg[i]),
            geometry.positions_enu[i],
            agl, gsd, east_min, north_max,
        )
        for i in range(len(images))
    ]
    canvas_w = int(np.ceil((east_max - east_min) / gsd)) + 1
    canvas_h = int(np.ceil((north_max - north_min) / gsd)) + 1

    print(
        f"[ortho] Canvas {canvas_w}x{canvas_h} px @ {gsd:.3f} m/px. AGL = {agl:.1f} m."
    )

    # best_score: distanza-dal-bordo del frame "vincente" per ciascun pixel.
    # Inizializzato a -1 → qualunque frame con score >= 0 vince al primo passaggio.
    best_score = np.full((canvas_h, canvas_w), -1.0, dtype=np.float32)
    best_img = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for i, img in enumerate(images):
        h, w = img.shape[:2]
        # Score sorgente = distanza dal bordo (uncapped). Massimo al centro,
        # cala verso 0 ai bordi del frame. Pixel fuori dal frame: 0.
        mask = np.ones((h, w), dtype=np.uint8)
        mask[:1, :] = 0
        mask[-1:, :] = 0
        mask[:, :1] = 0
        mask[:, -1:] = 0
        score = cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)

        M = M_use[i]

        warped_img = cv2.warpAffine(
            img, M, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        # NEAREST per lo score: evita aloni a valori bassi attorno ai bordi
        # warpati, che farebbero competere pixel "appena fuori" frame con
        # quelli interni di un frame vicino.
        warped_score = cv2.warpAffine(
            score, M, (canvas_w, canvas_h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )

        update = warped_score > best_score
        np.copyto(best_score, warped_score, where=update)
        np.copyto(best_img, warped_img, where=update[..., None])

        if (i + 1) % 20 == 0 or i == len(images) - 1:
            print(f"  ortho {i + 1}/{len(images)}")

    return OrthoResult(
        image=best_img,
        gsd=gsd,
        east_min=float(east_min),
        north_max=float(north_max),
        agl=float(agl),
    )
