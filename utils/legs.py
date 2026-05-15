"""Raggruppa i frame in leg (passate dritte del lawnmower) e gestisce il sottocampionamento.

Un leg = sequenza massimale di frame non-curva consecutivi. I frame curva fra leg N e
leg N+1 vengono assegnati come "bridge" geometrico a entrambi: partecipano al matching
per ancorare le passate adiacenti, ma non finiscono nel mosaico finale.
"""
import math


def group_into_legs(records: list[dict]) -> list[dict]:
    """Ritorna una lista di leg. Ogni leg e' un dict:
      - "frames": indici (in `records`) dei frame non-curva, in ordine temporale
      - "bridge_prev": indici dei frame curva immediatamente precedenti il leg
      - "bridge_next": indici dei frame curva immediatamente successivi al leg

    I bridge sono condivisi: gli stessi indici curva compaiono come bridge_next del leg
    precedente e come bridge_prev del leg successivo.
    """
    n = len(records)
    legs: list[dict] = []
    pending_curves: list[int] = []

    i = 0
    while i < n:
        if records[i]["is_curve"]:
            pending_curves.append(i)
            i += 1
        else:
            start = i
            while i < n and not records[i]["is_curve"]:
                i += 1
            legs.append(
                {
                    "frames": list(range(start, i)),
                    "bridge_prev": pending_curves,
                    "bridge_next": [],
                }
            )
            pending_curves = []

    for k in range(len(legs) - 1):
        legs[k]["bridge_next"] = legs[k + 1]["bridge_prev"]

    return legs


def subsample_leg_by_overlap(
    leg_positions: list[tuple[float, float]],
    image_height_px: int,
    gsd_m_per_px: float,
    target_frontal_overlap: float,
) -> list[int]:
    """Indici (relativi al leg) dei frame da tenere per garantire ~target_frontal_overlap.

    Tiene sempre il primo e l'ultimo del leg. In mezzo accumula spaziatura GPS e tiene
    un frame appena la distanza dal precedente "tenuto" raggiunge footprint*(1-overlap).
    """
    n = len(leg_positions)
    if n <= 2:
        return list(range(n))

    footprint_m = image_height_px * gsd_m_per_px
    min_step_m = footprint_m * max(1.0 - target_frontal_overlap, 0.05)

    kept = [0]
    last_x, last_y = leg_positions[0]
    for i in range(1, n - 1):
        x, y = leg_positions[i]
        if math.hypot(x - last_x, y - last_y) >= min_step_m:
            kept.append(i)
            last_x, last_y = x, y
    kept.append(n - 1)
    return kept
