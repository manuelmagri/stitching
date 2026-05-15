"""Sorgente dell'overlap: dichiarato dall'utente XOR dedotto dai dati GPS.

`overlap_source` deve essere "manual" o "auto" — qualunque altro valore alza un errore.
La modalita' "auto" stima un singolo (laterale, frontale) globale dal pattern del volo,
usando spacing intra-leg per il frontale e distanza fra leg adiacenti per il laterale.
"""
import math

import numpy as np


def compute_auto_overlap(
    legs: list[dict],
    positions_m: list[tuple[float, float]],
    image_size: tuple[int, int],
    gsds: list[float],
) -> tuple[float, float]:
    """Stima globale (lateral, frontal) in [0, 1).

    - frontal = 1 - spacing_intra_leg_mediano / (h_px * gsd_mediano)
    - lateral = 1 - spacing_inter_leg_mediano / (w_px * gsd_mediano)
    """
    w_px, h_px = image_size
    gsd_med = float(np.median(gsds)) if gsds else 1.0
    footprint_along = h_px * gsd_med
    footprint_cross = w_px * gsd_med

    intra_distances: list[float] = []
    for leg in legs:
        frames = leg["frames"]
        for k in range(len(frames) - 1):
            a, b = frames[k], frames[k + 1]
            dx = positions_m[a][0] - positions_m[b][0]
            dy = positions_m[a][1] - positions_m[b][1]
            intra_distances.append(math.hypot(dx, dy))
    frontal_spacing = float(np.median(intra_distances)) if intra_distances else 0.0
    frontal = max(0.0, min(0.99, 1.0 - frontal_spacing / max(footprint_along, 1e-9)))

    if len(legs) >= 2:
        centroids: list[tuple[float, float]] = []
        for leg in legs:
            frames = leg["frames"]
            cx = float(np.mean([positions_m[i][0] for i in frames]))
            cy = float(np.mean([positions_m[i][1] for i in frames]))
            centroids.append((cx, cy))
        inter_distances = [
            math.hypot(
                centroids[k + 1][0] - centroids[k][0],
                centroids[k + 1][1] - centroids[k][1],
            )
            for k in range(len(centroids) - 1)
        ]
        lateral_spacing = float(np.median(inter_distances))
        lateral = max(
            0.0, min(0.99, 1.0 - lateral_spacing / max(footprint_cross, 1e-9))
        )
    else:
        lateral = 0.0

    return lateral, frontal


def resolve_overlap_source(cfg: dict) -> str:
    """Valida cfg['overlap_source']. Solleva ValueError se non e' 'manual' o 'auto'."""
    src = cfg.get("overlap_source")
    if src not in ("manual", "auto"):
        raise ValueError(
            f"overlap_source non valido: {src!r}. Deve essere 'manual' oppure 'auto'."
        )
    return src
