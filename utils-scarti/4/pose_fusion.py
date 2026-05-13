"""Pose finali usate dal mosaico: posizione GPS + marcatura frame in curva.

Per la Fase 1 del nuovo mosaico la posizione di ciascun frame e' quella GPS
(in UTM relativo all'origine). Motivo: con sigma GPS ~1.4 m e scena planare
nadir, una VO 2D accumulata e' meno affidabile delle pose GPS stesse, soprattutto
intra-pass dove l'errore relativo GPS e' di decimetri.

Cio' che viene fuso qui:
  - posizione = GPS (utm_frames[i].east_rel, .north_rel);
  - yaw = flight_yaw IMU (passato attraverso, usato per il fallback nel mosaico
    quando il bearing GPS del rettilineo non e' disponibile);
  - flag `in_curve` calcolato dallo yaw rate IMU (deg/sec): sopra
    `curve_yaw_rate_deg_s` lo step e' considerato una virata.

Per il drift cross-strip residuo (sigma GPS-relativo ~m tra passate) si
prevede una Fase 2 con refinement geometrico ORB; l'API qui sotto e' stabile.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .gps_utm import UtmFrame
from .matching import MatchResult


@dataclass
class Pose:
    """Posa 2D di un frame usata dal mosaico."""
    x: float          # east_rel (m)
    y: float          # north_rel (m)
    yaw_deg: float    # flight_yaw IMU (compass deg)
    in_curve: bool    # True se lo yaw rate verso il frame supera la soglia


def fonde_traiettoria(
    matches_per_step: list[MatchResult | None],
    utm_frames: list[UtmFrame],
    camera_matrix: np.ndarray,
    timestamps: list[datetime],
    flight_yaws_deg: list[float],
    curve_yaw_rate_deg_s: float,
) -> list[Pose]:
    """Ritorna una `Pose` per frame.

    `matches_per_step` e `camera_matrix` non sono usati nella Fase 1
    (compatibilita' di firma per quando entrera' il refinement geometrico).
    """
    n = len(utm_frames)
    if n == 0:
        return []
    if len(flight_yaws_deg) != n or len(timestamps) != n:
        raise ValueError(
            f"Dimensioni: utm_frames={n}, yaws={len(flight_yaws_deg)}, "
            f"timestamps={len(timestamps)}"
        )

    poses: list[Pose] = [Pose(
        x=float(utm_frames[0].east_rel),
        y=float(utm_frames[0].north_rel),
        yaw_deg=float(flight_yaws_deg[0]),
        in_curve=False,
    )]

    for i in range(1, n):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
        dt = max(dt, 1e-3)
        dyaw = (flight_yaws_deg[i] - flight_yaws_deg[i - 1] + 180.0) % 360.0 - 180.0
        yaw_rate = abs(dyaw) / dt
        in_curve = yaw_rate > curve_yaw_rate_deg_s

        poses.append(Pose(
            x=float(utm_frames[i].east_rel),
            y=float(utm_frames[i].north_rel),
            yaw_deg=float(flight_yaws_deg[i]),
            in_curve=bool(in_curve),
        ))

    return poses


def indici_in_curva(poses: list[Pose]) -> list[int]:
    """Indici dei frame marcati come `in_curve` (per evidenziarli nel plot)."""
    return [i for i, p in enumerate(poses) if p.in_curve]
