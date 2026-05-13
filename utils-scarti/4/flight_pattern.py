"""Segmentazione del volo a greca in rettilinei e curve.

Il drone percorre rettilinei paralleli con virate (~180 deg) tra una passata
e l'altra. La rilevazione si basa sul bearing GPS dello step i->i+1: finche'
resta entro `delta_bearing_max_deg` dall'ancora del rettilineo corrente, lo
step appartiene allo stesso rettilineo; quando cambia significativamente,
chiude il rettilineo corrente, marca i frame della virata come transizione,
e apre un nuovo rettilineo al primo edge stabile successivo.

Convenzione bearing: compass-deg, 0 = Nord, +90 = Est, in [0, 360).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gps_utm import UtmFrame


def _bearing_compass_deg(dE: float, dN: float) -> float:
    """Bearing in compass-degrees (0=N, +90=E) per uno step (dE, dN) in UTM."""
    return float((np.degrees(np.arctan2(dE, dN)) + 360.0) % 360.0)


def _wrap_diff_deg(a: float, b: float) -> float:
    """(a - b) wrapped in [-180, 180]."""
    return float((a - b + 180.0) % 360.0 - 180.0)


@dataclass
class FlightStructure:
    """Volo decomposto in una sequenza di rettilinei e frame di transizione."""
    rows: list[list[int]]                # indici dei frame per ciascun rettilineo
    row_bearings_deg: list[float]        # bearing medio (compass) di ciascun rettilineo
    transition_frames: list[int]         # frame esclusi dai rettilinei (curve)

    def descrivi(self) -> str:
        if not self.rows:
            return "nessun rettilineo rilevato"
        parti = [
            f"r{r}: {len(idx)} frame, bearing {self.row_bearings_deg[r]:.1f} deg"
            for r, idx in enumerate(self.rows)
        ]
        if self.transition_frames:
            parti.append(f"transizioni: {len(self.transition_frames)} frame")
        return "; ".join(parti)


def rileva_rettilinei(utm_frames: list[UtmFrame],
                      delta_bearing_max_deg: float = 25.0,
                      min_passo_m: float = 0.3,
                      min_frame_per_rettilineo: int = 3) -> FlightStructure:
    """Segmenta i frame in rettilinei sulla base del bearing GPS edge-by-edge."""
    n = len(utm_frames)
    if n < 2:
        return FlightStructure(rows=[], row_bearings_deg=[], transition_frames=[])

    bearings = np.zeros(n - 1, dtype=np.float64)
    steps = np.zeros(n - 1, dtype=np.float64)
    for i in range(n - 1):
        dE = utm_frames[i + 1].easting - utm_frames[i].easting
        dN = utm_frames[i + 1].northing - utm_frames[i].northing
        steps[i] = float(np.hypot(dE, dN))
        bearings[i] = _bearing_compass_deg(dE, dN)

    rows: list[list[int]] = []
    row_bearings: list[float] = []
    transitions: list[int] = []

    in_row = False
    current: list[int] = []
    anchor = 0.0
    sin_sum = cos_sum = 0.0

    def flush() -> None:
        nonlocal in_row, current, sin_sum, cos_sum
        if len(current) >= min_frame_per_rettilineo:
            rows.append(current[:])
            mean_b = float((np.degrees(np.arctan2(sin_sum, cos_sum)) + 360.0) % 360.0)
            row_bearings.append(mean_b)
        else:
            transitions.extend(current)
        current = []
        sin_sum = 0.0
        cos_sum = 0.0
        in_row = False

    for i in range(n - 1):
        if steps[i] < min_passo_m:
            # Quasi fermo: bearing rumoroso, tratta come transizione.
            if in_row:
                # Il frame i+1 (endpoint dello step quasi-zero) e' transitorio.
                transitions.append(i + 1)
            else:
                transitions.extend([i, i + 1])
            continue

        if not in_row:
            in_row = True
            anchor = bearings[i]
            current = [i, i + 1]
            rad = np.radians(bearings[i])
            sin_sum = float(np.sin(rad))
            cos_sum = float(np.cos(rad))
            continue

        if abs(_wrap_diff_deg(bearings[i], anchor)) <= delta_bearing_max_deg:
            current.append(i + 1)
            rad = np.radians(bearings[i])
            sin_sum += float(np.sin(rad))
            cos_sum += float(np.cos(rad))
        else:
            # Cambio di bearing: chiudi e marca i+1 come transizione (e' nella virata).
            flush()
            transitions.append(i + 1)

    if in_row:
        flush()

    # Frame che compaiono anche in qualche rettilineo non sono transizione.
    in_row_set = {i for row in rows for i in row}
    transitions = sorted({t for t in transitions if t not in in_row_set})

    return FlightStructure(
        rows=rows,
        row_bearings_deg=row_bearings,
        transition_frames=transitions,
    )


def ordine_warp(struct: FlightStructure) -> list[int]:
    """Ordine di warp del mosaico: concatenazione dei rettilinei, esclude le curve.

    I frame in `transition_frames` non vengono warpati: la loro omografia
    analitica (yaw=bearing del rettilineo) sarebbe arbitraria, e fanno
    comparire artefatti rotazionali al canvas.
    """
    out: list[int] = []
    for row in struct.rows:
        out.extend(row)
    return out


def stima_passi_metrici(struct: FlightStructure,
                        utm_frames: list[UtmFrame]) -> tuple[float | None, float | None]:
    """Stima il passo frontale (entro rettilineo) e laterale (tra rettilinei adiacenti).

    Frontale: mediana di |delta posizione| tra frame consecutivi di ogni rettilineo.
    Laterale: per ogni coppia (r, r+1), mediana della distanza dei frame di r+1
    dal punto medio di r, proiettata sulla normale al bearing di r.
    Ritorna None per il laterale se i rettilinei sono meno di 2.
    """
    fronts: list[float] = []
    for row in struct.rows:
        for j in range(1, len(row)):
            a, b = row[j - 1], row[j]
            fronts.append(float(np.hypot(
                utm_frames[b].easting - utm_frames[a].easting,
                utm_frames[b].northing - utm_frames[a].northing,
            )))
    passo_front = float(np.median(fronts)) if fronts else None

    laterals: list[float] = []
    for r in range(len(struct.rows) - 1):
        row_a = struct.rows[r]
        row_b = struct.rows[r + 1]
        # Vettore direzione del rettilineo r (compass bearing -> ENU unit):
        #   forward = (sin(b), cos(b))      (per b=0 -> Nord; per b=90 -> Est)
        #   normal  = (cos(b), -sin(b))     (90 deg a destra del forward)
        b_r = np.radians(struct.row_bearings_deg[r])
        nx = float(np.cos(b_r))
        ny = -float(np.sin(b_r))
        cx = float(np.mean([utm_frames[i].easting for i in row_a]))
        cy = float(np.mean([utm_frames[i].northing for i in row_a]))
        ds = []
        for i in row_b:
            dx = utm_frames[i].easting - cx
            dy = utm_frames[i].northing - cy
            ds.append(abs(dx * nx + dy * ny))
        if ds:
            laterals.append(float(np.median(ds)))
    passo_lat = float(np.median(laterals)) if laterals else None

    return passo_front, passo_lat
