"""Frame locale metrico, costruito senza GPS.

Il GPS non entra nello stitching: serve solo alla georeferenziazione finale
(`utils.georeference`). Qui il riferimento e' costruito dai soli sensori di bordo:

    scala       <- quota barometrica (XMP:RelativeAltitude) e focale, via GSD = quota / focale
    rotazione   <- bussola (XMP:FlightYawDegree), quindi riferita al NORD VERO
    traslazione <- data/translations.json, velocita' XMP integrate sul tempo fra scatti

L'origine e' il primo frame selezionato. Restano indeterminati i soli due gradi di
liberta' della posizione assoluta, che sono esattamente cio' che il GPS fornisce alla fine.

Due avvertenze misurate sul volo di prova, entrambe rilevanti per chi legge queste
posizioni:

- L'odometria e' un SEED, non una verita'. Tolti rototraslazione e scala, il suo errore
  di forma contro il GPS e' 4,2 m RMS con punte di 11,7 m. Localmente pero' e' ottima
  (0,46 m per passo, ~0,6 m accumulati fra due passate adiacenti), ed e' la scala locale
  quella che conta per decidere quali immagini si sovrappongono. A raddrizzare la forma
  ci pensano i vincoli fotografici in `utils.poses`.

- La bussola misura il nord VERO, UTM usa il nord GRIGLIA. Al sito di prova differiscono
  di 1,073 gradi, cioe' 5,7 m sui 297 m del volo. La differenza viene assorbita dal fit
  di similarita' di `utils.georeference`, che per questo deve includere la rotazione.
"""
import json
from pathlib import Path

import numpy as np

from utils.dataset import altitude_m
from utils.geodesy import gsd_meters_per_pixel


def load_vo_deltas(translations_path: Path) -> np.ndarray:
    """Delta inter-frame in metri, array (N, 2) nell'ordine (est, nord).

    Il file e' prodotto da `preprocessing.create_translations`, che ha gia' scambiato gli
    assi XMP DJI (dove FlightXSpeed e' la componente NORD e FlightYSpeed quella EST) e
    moltiplicato per il tempo fra scatti. Qui non resta nessuna conversione da fare.
    """
    with open(translations_path) as f:
        data = json.load(f)

    deltas = np.asarray(data, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] != 2:
        raise ValueError(
            f"{translations_path} dovrebbe contenere una lista di coppie [est, nord], "
            f"trovato un array di forma {deltas.shape}"
        )
    return deltas


def integrate_positions(vo_deltas: np.ndarray) -> np.ndarray:
    """Posizioni assolute (N, 2) in metri, integrando i delta a partire da (0, 0).

    Il primo delta viene scartato: non esiste uno scatto precedente da cui misurarlo, e
    `create_translations` gli assegna convenzionalmente 1 s di velocita' al decollo.
    """
    if len(vo_deltas) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    passi = np.vstack([np.zeros((1, 2)), vo_deltas[1:]])
    return np.cumsum(passi, axis=0)


def positions_for(records: list[dict], positions_all: np.ndarray) -> np.ndarray:
    """Posizioni (M, 2) dei soli `records`, riportate all'origine sul primo di essi.

    Indirizza `positions_all` tramite `vo_idx`, che e' l'indice nella lista COMPLETA dei
    record: cosi' un filtro sul range di frame non disallinea l'odometria.
    """
    if not records:
        return np.zeros((0, 2), dtype=np.float64)

    idx = [r["vo_idx"] for r in records]
    fuori = [i for i in idx if not 0 <= i < len(positions_all)]
    if fuori:
        raise ValueError(
            f"vo_idx fuori dall'odometria ({len(positions_all)} righe): {fuori[:5]}. "
            "metadata.json e translations.json non descrivono lo stesso volo."
        )

    posizioni = positions_all[idx]
    return posizioni - posizioni[0]


def frame_gsds(records: list[dict], focal_px: float) -> np.ndarray:
    """GSD in m/px di ogni record, dalla quota barometrica e dalla focale rettificata."""
    return np.array(
        [gsd_meters_per_pixel(altitude_m(r), focal_px) for r in records], dtype=np.float64
    )
