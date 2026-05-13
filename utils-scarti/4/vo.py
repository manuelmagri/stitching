"""Visual odometry 2D approssimata, utile solo a scopo di visualizzazione.

Approccio leggero, pensato per il confronto qualitativo con la traiettoria
GPS nel plot HTML, non per essere usato come posa del mosaico:

  1. Per ogni step i->i+1 con match RANSAC validi, calcola lo shift mediano
     in pixel tra inlier (q2 - q1).
  2. Stima un fattore globale m_per_pixel = mediana( |Delta-GPS| / |shift_px| )
     su tutte le coppie "stabili" (yaw quasi fermo, drone in movimento).
  3. Converte ogni shift pixel in spostamento metrico mondo applicando
     m_per_pixel e ruotando con il flight_yaw del frame precedente.
  4. Accumula partendo dalla posizione GPS del primo scatto.

Per i pose del mosaico si usa direttamente la posa GPS (vedi `pose_fusion`):
la VO accumulata su scena planare nadir tende a derivare ed e' meno affidabile
del GPS sigma ~1.4 m sul caso reale.
"""
from __future__ import annotations

import numpy as np

from .gps_utm import UtmFrame
from .matching import MatchResult


def _global_m_per_pixel(matches_per_step: list[MatchResult | None],
                        utm_frames: list[UtmFrame],
                        flight_yaws_deg: list[float],
                        yaw_stability_deg: float,
                        min_displacement_m: float,
                        min_inliers: int) -> float | None:
    """Mediana del rapporto |Delta GPS| / |shift_px| su coppie stabili."""
    n = len(utm_frames)
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
        gps_m = float(np.hypot(dE, dN))
        if gps_m < min_displacement_m:
            continue
        shift = np.median(mr.q2 - mr.q1, axis=0)
        px = float(np.hypot(shift[0], shift[1]))
        if px > 1.0:
            ratios.append(gps_m / px)
    if not ratios:
        return None
    return float(np.median(ratios))


def stima_traiettoria_vo_2d(
    matches_per_step: list[MatchResult | None],
    utm_frames: list[UtmFrame],
    flight_yaws_deg: list[float],
    *,
    yaw_stability_deg: float = 2.0,
    min_displacement_m: float = 0.5,
    min_inliers: int = 12,
) -> list[tuple[float, float]]:
    """Traiettoria 2D VO (m, riferita all'origine del primo scatto), per il plot.

    Quando i match di una step sono insufficienti, ricade sul Delta-GPS dello
    step come fallback (cosi' la traiettoria resta continua).
    """
    n = len(utm_frames)
    if n == 0:
        return []
    if n == 1:
        return [(utm_frames[0].east_rel, utm_frames[0].north_rel)]

    m_per_px = _global_m_per_pixel(
        matches_per_step, utm_frames, flight_yaws_deg,
        yaw_stability_deg, min_displacement_m, min_inliers,
    )

    x = float(utm_frames[0].east_rel)
    y = float(utm_frames[0].north_rel)
    path: list[tuple[float, float]] = [(x, y)]

    for i in range(1, n):
        mr = matches_per_step[i - 1] if i - 1 < len(matches_per_step) else None
        dE_gps = utm_frames[i].east_rel - utm_frames[i - 1].east_rel
        dN_gps = utm_frames[i].north_rel - utm_frames[i - 1].north_rel

        if m_per_px is None or mr is None or len(mr) < min_inliers:
            x += dE_gps
            y += dN_gps
            path.append((x, y))
            continue

        shift = np.median(mr.q2 - mr.q1, axis=0)
        su_m = float(shift[0]) * m_per_px
        sv_m = float(shift[1]) * m_per_px

        # Mappa lo shift pixel mediano in coord mondo via il flight_yaw del
        # frame i-1 (compass deg). La camera si muove all'opposto rispetto ai
        # punti dell'immagine. Convenzione coerente con homography_curve.py:
        #   pixel (du, dv)  -> world  ( gsd*c*du - gsd*s*dv,  -gsd*s*du - gsd*c*dv )
        # con c = cos(yaw), s = sin(yaw). Lo step della camera = -step_punti.
        yaw_r = np.radians(flight_yaws_deg[i - 1])
        c, s = float(np.cos(yaw_r)), float(np.sin(yaw_r))
        world_dE_points = c * su_m - s * sv_m
        world_dN_points = -s * su_m - c * sv_m
        cam_dE = -world_dE_points
        cam_dN = -world_dN_points

        # Ri-scala il passo VO al |Delta GPS| (rimuove deriva di scala globale).
        gps_step = float(np.hypot(dE_gps, dN_gps))
        vo_step = float(np.hypot(cam_dE, cam_dN))
        if vo_step > 0.05 and gps_step > 0.05:
            scale = gps_step / vo_step
            cam_dE *= scale
            cam_dN *= scale
            x += cam_dE
            y += cam_dN
        else:
            x += dE_gps
            y += dN_gps
        path.append((x, y))

    return path
