"""Parser dei delta di odometria visiva da translations.txt.

translations.txt contiene un vettore per riga del tipo:
    Translation Vector N: [x y z]

Verificato sperimentalmente confrontando direzione VO con flight_yaw EXIF:
- **Asse 0 del file = NORD, asse 1 = EST** (non e' ENU standard!). Lo swap viene
  fatto qui al load, cosi' tutti i caller ricevono (east, north).
- I valori sono inter-frame in metri (ogni riga = delta da frame N-1 a frame N).
- Senza alignment, la traiettoria VO swappata aderisce al GPS con RMSE ~6 m
  (entro il rumore GPS); senza swap il RMSE schizza a ~180 m → swap obbligatorio.
- `scales.txt` NON vanno applicati alle traslazioni (sono gia' in metri).

trans[0] e' la prima riga del file (Translation Vector 1) ed e' praticamente nulla,
perche' non c'e' un frame precedente. Per k >= 1, trans[k] = delta da records_all[k-1]
a records_all[k].
"""
import re
from pathlib import Path

import numpy as np


def load_vo_deltas(path: Path) -> np.ndarray:
    """Ritorna un array (N, 2) coi delta inter-frame in metri ENU (Est, Nord).

    Il file ha l'asse 0 = NORD, asse 1 = EST. Qui swappiamo le colonne cosi' il
    risultato e' nell'ordine (east, north) usato dal resto della pipeline.
    """
    pat = re.compile(r"\[([^\]]*)\]")
    out: list[list[float]] = []
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        vals = [float(v) for v in m.group(1).split() if v]
        if len(vals) >= 2:
            # File: vals[0]=north, vals[1]=east -> restituiamo (east, north)
            out.append([vals[1], vals[0]])
    return np.array(out, dtype=np.float64)


def cumulative_delta_m(
    vo_deltas_m: np.ndarray, vo_idx_from: int, vo_idx_to: int
) -> tuple[float, float]:
    """Somma cumulativa delta da records_all[vo_idx_from] a records_all[vo_idx_to].

    Tipicamente vo_idx_to > vo_idx_from; il segno gestisce il verso inverso.
    Restituisce (delta_east_m, delta_north_m).
    """
    if vo_idx_to == vo_idx_from:
        return 0.0, 0.0
    a, b = sorted([vo_idx_from, vo_idx_to])
    seg = vo_deltas_m[a + 1 : b + 1]
    delta = seg.sum(axis=0) if len(seg) else np.zeros(2)
    if vo_idx_to < vo_idx_from:
        delta = -delta
    return float(delta[0]), float(delta[1])
