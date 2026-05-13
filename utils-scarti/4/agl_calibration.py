"""Auto-calibrazione AGL: stima l'altezza sul terreno dal rapporto
Delta-posizione GPS / Delta-shift pixel.

L'AGL EXIF (Relative Altitude di DJI) e' rispetto al punto di decollo: se il
terreno sotto il drone e' a quota diversa rispetto al punto di decollo, l'AGL
EXIF e' sbagliato e la GSD del mosaico ne risente direttamente. Stimarla dai
match feature e' robusto: si confronta lo spostamento metrico GPS con lo
spostamento pixel mediano nei match RANSAC-inlier.

Filtri applicati a ciascuna coppia di frame consecutivi:
  - yaw quasi stabile (|Delta yaw| < `yaw_stability_deg`): elimina i frame in
    curva, dove pure traslazione planare non si applica;
  - drone in movimento (|Delta posizione GPS| > `min_displacement_m`): elimina
    gli scatti quasi fermi, dove il rapporto e' rumoroso;
  - abbastanza match RANSAC-inlier (`min_inliers`).

Aggregazione: mediana dei rapporti m/pixel sulle coppie valide (robusta agli
outlier per singoli match cattivi).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gps_utm import UtmFrame
from .matching import MatchResult


@dataclass
class AglEstimate:
    """Risultato dell'auto-calibrazione AGL."""
    agl_m: float            # altezza stimata (m)
    m_per_pixel: float      # GSD alla risoluzione di lavoro (m/px)
    n_pairs_used: int       # numero di coppie consecutive aggregate
    relative_std: float     # std relativa dei rapporti (proxy di affidabilita')


def auto_calibrate_agl(
    matches_per_step: list[MatchResult | None],
    utm_frames: list[UtmFrame],
    flight_yaws_deg: list[float],
    focal_px_scaled: float,
    *,
    yaw_stability_deg: float = 2.0,
    min_displacement_m: float = 1.0,
    min_inliers: int = 20,
    max_pairs: int = 60,
) -> AglEstimate | None:
    """Stima AGL dai match feature + spostamenti GPS.

    `matches_per_step[i-1]` e' il MatchResult tra il frame i-1 e il frame i, gia'
    filtrato RANSAC dal `match_features`. Lo shift pixel mediano sui suoi inlier
    e' la traslazione apparente della scena. Usando la focale alla risoluzione
    di lavoro: AGL = (gps_disp_m / pixel_disp) * focal_px_scaled.

    Ritorna None se le coppie valide sono meno di 3.
    """
    if not matches_per_step:
        return None
    n = len(utm_frames)
    if len(flight_yaws_deg) != n or len(matches_per_step) != n - 1:
        raise ValueError(
            f"Dimensioni disallineate: utm_frames={n}, flight_yaws={len(flight_yaws_deg)}, "
            f"matches_per_step={len(matches_per_step)} (atteso {n - 1})"
        )

    ratios: list[float] = []
    for i in range(1, n):
        mr = matches_per_step[i - 1]
        if mr is None or len(mr) < min_inliers:
            continue

        dyaw = (flight_yaws_deg[i] - flight_yaws_deg[i - 1] + 180.0) % 360.0 - 180.0
        if abs(dyaw) > yaw_stability_deg:
            continue

        dE = utm_frames[i].easting - utm_frames[i - 1].easting
        dN = utm_frames[i].northing - utm_frames[i - 1].northing
        gps_disp_m = float(np.hypot(dE, dN))
        if gps_disp_m < min_displacement_m:
            continue

        # Shift mediano in pixel sui soli inlier RANSAC.
        shift = np.median(mr.q2 - mr.q1, axis=0)
        pixel_disp = float(np.hypot(shift[0], shift[1]))
        if pixel_disp < 1.0:
            continue

        ratios.append(gps_disp_m / pixel_disp)
        if len(ratios) >= max_pairs:
            break

    if len(ratios) < 3:
        return None

    m_per_pixel = float(np.median(ratios))
    agl_m = m_per_pixel * float(focal_px_scaled)
    rel_std = float(np.std(ratios) / m_per_pixel) if m_per_pixel > 0 else float("inf")

    return AglEstimate(
        agl_m=agl_m,
        m_per_pixel=m_per_pixel,
        n_pairs_used=len(ratios),
        relative_std=rel_std,
    )


def confronta_overlap(
    agl_m: float,
    camera_matrix_scaled: np.ndarray,
    step_m_mediano: float,
    img_w_px: int,
    img_h_px: int,
) -> tuple[float, float]:
    """Calcola overlap frontale e laterale dato AGL + passo medio GPS.

    Footprint metrico per asse:
        fp_v = AGL * h_px / fy  (asse v dell'immagine = direzione di volo)
        fp_u = AGL * w_px / fx  (asse u = perpendicolare)
    Overlap implicato dal passo:
        overlap = 1 - step_m / footprint

    Si assume lo stesso passo (frontale) per entrambi gli assi: e' un'approssimazione
    utile a confrontare con OVERLAP_FRONTAL dichiarato. Per il laterale reale
    serve la stima di `stima_passi_metrici`.
    """
    fx = float(camera_matrix_scaled[0, 0])
    fy = float(camera_matrix_scaled[1, 1])
    fp_v = agl_m * float(img_h_px) / max(fy, 1e-9)
    fp_u = agl_m * float(img_w_px) / max(fx, 1e-9)
    ovl_front = 1.0 - step_m_mediano / max(fp_v, 1e-9)
    ovl_lat = 1.0 - step_m_mediano / max(fp_u, 1e-9)
    return float(ovl_front), float(ovl_lat)
