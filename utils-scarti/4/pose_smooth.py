"""Smoothing temporale leggero delle posizioni GPS + sanity check IMU.

Le posizioni GPS hanno un jitter alta frequenza (anche con RTK fisso); peggio
ancora se l'RTK e' in modalita' Single (caso del dataset corrente, sigma ~1.4 m).
Frame consecutivi hanno errore correlato, ma sub-frame restano oscillazioni
visibili che, propagate nel mosaico, producono micro-gradini tra tile adiacenti.

Approccio: kernel gaussiano nel tempo (sigma piccola, ~0.7 s) sui punti UTM.
Riduce il jitter sub-secondo senza intaccare la macro-struttura della traiettoria
(rettilinei + curve). Non e' un filtro di Kalman: niente modello dinamico.

Inoltre offre un sanity check: confronto fra velocita' GPS-da-timestamp e
velocita' IMU (FlightXSpeed/YSpeed). Discrepanze > soglia indicano salti GPS
o timestamp anomali; si emette la lista degli indici sospetti per uso a valle.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .gps_utm import UtmFrame, UtmProjector


@dataclass
class SmoothingResult:
    """Risultato di `liscia_traiettoria_gps`."""
    utm_frames: list[UtmFrame]      # frame con posizioni UTM lisciate
    jitter_rms_m: float             # RMS dello spostamento applicato (m)


def _gauss_smooth_xy(t: np.ndarray, x: np.ndarray, y: np.ndarray,
                     sigma_t_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Media pesata gaussiana di (x, y) in funzione del tempo `t` (secondi)."""
    n = len(t)
    xs = np.empty(n)
    ys = np.empty(n)
    inv_two_sigma_sq = 1.0 / (2.0 * sigma_t_s * sigma_t_s)
    for i in range(n):
        dt = t - t[i]
        w = np.exp(-(dt * dt) * inv_two_sigma_sq)
        ws = w.sum()
        xs[i] = (w * x).sum() / ws
        ys[i] = (w * y).sum() / ws
    return xs, ys


def liscia_traiettoria_gps(
    utm_frames: list[UtmFrame],
    timestamps: list[datetime],
    projector: UtmProjector,
    *,
    sigma_t_s: float = 0.7,
) -> SmoothingResult:
    """Smoothing gauss temporale (sigma=0.7 s di default) delle posizioni UTM.

    Il sigma e' volutamente piccolo: a ~1 Hz di acquisizione equivale a una
    media leggera tra frame i e i suoi 1-2 vicini, sufficiente a smorzare il
    jitter senza distruggere le curve.
    """
    n = len(utm_frames)
    if n < 3 or sigma_t_s <= 0:
        return SmoothingResult(utm_frames=list(utm_frames), jitter_rms_m=0.0)
    if len(timestamps) != n:
        raise ValueError(
            f"timestamps ({len(timestamps)}) deve avere stessa lunghezza di utm_frames ({n})"
        )

    t = np.array([ts.timestamp() for ts in timestamps], dtype=np.float64)
    E = np.array([f.easting for f in utm_frames], dtype=np.float64)
    N = np.array([f.northing for f in utm_frames], dtype=np.float64)

    E_s, N_s = _gauss_smooth_xy(t, E, N, sigma_t_s)
    jitter = float(np.sqrt(np.mean((E - E_s) ** 2 + (N - N_s) ** 2)))

    east0 = float(projector.east0)
    north0 = float(projector.north0)
    smoothed = [
        UtmFrame(
            easting=float(E_s[i]),
            northing=float(N_s[i]),
            alt_m=utm_frames[i].alt_m,
            east_rel=float(E_s[i] - east0),
            north_rel=float(N_s[i] - north0),
        )
        for i in range(n)
    ]
    return SmoothingResult(utm_frames=smoothed, jitter_rms_m=jitter)


def rileva_outlier_gps_vs_imu(
    utm_frames: list[UtmFrame],
    timestamps: list[datetime],
    flight_speeds: list[np.ndarray],
    *,
    max_discrepancy_m_s: float = 3.0,
) -> list[int]:
    """Indici dei frame con discrepanza |v_gps - v_imu| > soglia.

    v_gps[i] = |posizione[i] - posizione[i-1]| / dt   (m/s)
    v_imu[i] = ||(FlightXSpeed, FlightYSpeed)||       (m/s)

    Soglia di default 3 m/s: a 1 Hz e velocita' nominale ~3-5 m/s, qualsiasi
    salto > 3 m/s sopra il previsto e' o un salto GPS o un timestamp duplicato.
    """
    n = len(utm_frames)
    if n < 2:
        return []
    if len(timestamps) != n or len(flight_speeds) != n:
        raise ValueError("Dimensioni disallineate fra utm_frames, timestamps, flight_speeds")

    out: list[int] = []
    for i in range(1, n):
        dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
        dt = max(dt, 1e-3)
        dE = utm_frames[i].easting - utm_frames[i - 1].easting
        dN = utm_frames[i].northing - utm_frames[i - 1].northing
        v_gps = float(np.hypot(dE, dN) / dt)
        sp = flight_speeds[i]
        v_imu = float(np.hypot(sp[0], sp[1]))
        if abs(v_gps - v_imu) > max_discrepancy_m_s:
            out.append(i)
    return out
