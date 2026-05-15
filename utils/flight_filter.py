"""Marca i frame acquisiti in curva (senza scartarli).

Un frame e' considerato "in curva" se almeno uno tra:
- |FlightRollDegree| supera roll_threshold_deg (drone inclinato durante la virata)
- |delta FlightYawDegree| verso il vicino temporale supera yaw_rate_threshold_deg

I frame curva non entrano nel mosaico finale ma restano disponibili come "bridge"
geometrici per agganciare passate adiacenti.
"""


def _yaw_diff(a: float, b: float) -> float:
    """Differenza minima fra due angoli in gradi, normalizzata a [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def mark_curves(
    records: list[dict],
    roll_threshold_deg: float = 8.0,
    yaw_rate_threshold_deg: float = 4.0,
) -> dict:
    """Aggiunge in-place il campo is_curve: bool a ogni record. Ritorna statistiche."""
    n = len(records)
    if n < 3:
        for r in records:
            r["is_curve"] = False
        return {"curve": 0, "straight": n, "curve_indices": []}

    for i, r in enumerate(records):
        if abs(r.get("flight_roll_deg", 0.0)) > roll_threshold_deg:
            r["is_curve"] = True
            continue
        prev_dy = (
            abs(_yaw_diff(r["flight_yaw_deg"], records[i - 1]["flight_yaw_deg"]))
            if i > 0
            else 0.0
        )
        next_dy = (
            abs(_yaw_diff(records[i + 1]["flight_yaw_deg"], r["flight_yaw_deg"]))
            if i < n - 1
            else 0.0
        )
        r["is_curve"] = max(prev_dy, next_dy) > yaw_rate_threshold_deg

    curve_indices = [r["index"] for r in records if r["is_curve"]]
    return {
        "curve": len(curve_indices),
        "straight": n - len(curve_indices),
        "curve_indices": curve_indices,
    }
