"""Segmentazione del percorso "alla greca" in passate rettilinee.

Una passata = serie di frame con heading approssimativamente costante. Tra
due passate adiacenti c'e' una virata (U-turn ~ 180 deg).
"""
from __future__ import annotations

import numpy as np


def unwrap_deg(yaws_deg: np.ndarray) -> np.ndarray:
    """Sblocca lo yaw evitando i salti +/-180 deg per ottenere una serie continua."""
    return np.degrees(np.unwrap(np.radians(np.asarray(yaws_deg, dtype=np.float64))))


def segment_passes(
    yaws_deg: np.ndarray,
    turn_thresh_deg: float = 90.0,
    min_pass_len: int = 5,
) -> list[tuple[int, int]]:
    """Segmenta gli indici frame in passate basandosi sull'evoluzione dello yaw.

    Algoritmo: percorre i frame e fa scattare una nuova passata quando l'heading
    si discosta dall'ancora corrente di piu' di `turn_thresh_deg`. Dopo lo split,
    l'ancora viene riassegnata al nuovo heading.

    Ritorna una lista di tuple (start, end_exclusive).
    """
    y = unwrap_deg(yaws_deg)
    n = len(y)
    if n == 0:
        return []
    if n <= min_pass_len:
        return [(0, n)]

    splits = [0]
    anchor = float(y[0])
    for i in range(1, n):
        if abs(float(y[i]) - anchor) > turn_thresh_deg and (i - splits[-1]) >= min_pass_len:
            splits.append(i)
            anchor = float(y[i])
    if splits[-1] != n:
        splits.append(n)

    passes: list[tuple[int, int]] = []
    for k in range(len(splits) - 1):
        s, e = splits[k], splits[k + 1]
        if e - s >= min_pass_len:
            passes.append((s, e))
    if not passes:
        passes = [(0, n)]
    return passes


def pass_mean_heading(yaws_deg: np.ndarray, pass_range: tuple[int, int]) -> float:
    """Heading medio (deg, in [-180, 180]) di una passata.

    Usa la media circolare (somma di vettori unitari) per essere robusto al
    wrap +/-180.
    """
    s, e = pass_range
    a = np.radians(np.asarray(yaws_deg[s:e], dtype=np.float64))
    mean = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
    return float(np.degrees(mean))
