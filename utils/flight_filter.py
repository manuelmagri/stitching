"""Filtra i frame acquisiti in curva.

Un frame e' considerato "in curva" se almeno uno tra:
- |FlightRollDegree| supera roll_threshold_deg (drone inclinato durante la virata)
- |delta FlightYawDegree| verso il vicino temporale supera yaw_rate_threshold_deg
"""


def _yaw_diff(a: float, b: float) -> float:
    """Differenza minima fra due angoli in gradi, normalizzata a [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def filter_curves(
    records: list[dict],
    roll_threshold_deg: float = 8.0,
    yaw_rate_threshold_deg: float = 4.0,
) -> tuple[list[dict], dict]:
    """Ritorna (record_filtrati, stats).

    stats contiene: dropped, kept, dropped_indices (i numeri di frame scartati).
    """
    n = len(records)
    if n < 3:
        return list(records), {"dropped": 0, "kept": n, "dropped_indices": []}

    is_curve = [False] * n
    for i, r in enumerate(records):
        if abs(r.get("flight_roll_deg", 0.0)) > roll_threshold_deg:
            is_curve[i] = True
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
        if max(prev_dy, next_dy) > yaw_rate_threshold_deg:
            is_curve[i] = True

    kept = [r for r, c in zip(records, is_curve) if not c]
    dropped_indices = [r["index"] for r, c in zip(records, is_curve) if c]
    return kept, {
        "dropped": len(dropped_indices),
        "kept": len(kept),
        "dropped_indices": dropped_indices,
    }
