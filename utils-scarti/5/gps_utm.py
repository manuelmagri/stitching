"""Conversione coordinate WGS84 (lat, lon) -> UTM e differenze metriche.

La pipeline lavora in un riferimento metrico locale: tutte le pose stimate e
quelle GPS vivono in metri UTM, con origine traslata sul primo scatto. Cosi'
il confronto delle traiettorie e il mosaico georeferenziato sono coerenti.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from pyproj import CRS, Transformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyproj non installato. Aggiungi `pyproj` a requirements.txt e installa."
    ) from exc


@dataclass
class UtmFrame:
    """Punto GPS in coordinate UTM, sia assolute che relative al primo scatto."""

    easting: float       # m, UTM
    northing: float      # m, UTM
    alt_m: float         # m s.l.m. (passa attraverso senza riproiezione)
    east_rel: float      # m rispetto all'origine
    north_rel: float     # m rispetto all'origine


def _utm_zone(lon_deg: float, lat_deg: float) -> int:
    """Numero di zona UTM (1..60) dalla longitudine."""
    # +180 per portare in [0, 360); zona = floor(lon/6) + 1.
    zone = int((lon_deg + 180.0) // 6.0) + 1
    return max(1, min(60, zone))


def _utm_epsg(lon_deg: float, lat_deg: float) -> int:
    """EPSG dell'UTM corrispondente: 326## nord, 327## sud."""
    zone = _utm_zone(lon_deg, lat_deg)
    return (32600 if lat_deg >= 0 else 32700) + zone


class UtmProjector:
    """Wrapper attorno a pyproj.Transformer per WGS84 <-> UTM.

    L'EPSG UTM viene scelto una volta sola dal primo punto fornito, evitando
    discontinuita' al confine di zona durante missioni brevi.
    """

    def __init__(self, lat0_deg: float, lon0_deg: float):
        self.epsg = _utm_epsg(lon0_deg, lat0_deg)
        self.crs = CRS.from_epsg(self.epsg)
        self._fwd = Transformer.from_crs(
            "EPSG:4326", self.crs, always_xy=True
        )
        # Origine UTM = posizione del primo punto, salvata per traslare in
        # coordinate locali "_rel".
        self.east0, self.north0 = self._fwd.transform(lon0_deg, lat0_deg)

    def to_utm(self, lat_deg: float, lon_deg: float):
        """Restituisce (easting, northing) in metri UTM (assoluti)."""
        return self._fwd.transform(lon_deg, lat_deg)

    def to_relative(self, lat_deg: float, lon_deg: float):
        """Coordinate metriche relative all'origine (primo scatto). (E, N)."""
        e, n = self.to_utm(lat_deg, lon_deg)
        return e - self.east0, n - self.north0


def proietta_frames(frames) -> tuple[UtmProjector, list[UtmFrame]]:
    """Costruisce il proiettore dal primo frame e proietta tutta la sequenza.

    `frames` e' una lista di FrameMeta (vedi io_loader). Ritorna il proiettore
    (per riusarlo a valle, es. nel writer GeoTIFF) e i punti UTM.
    """
    if not frames:
        raise ValueError("Lista frames vuota")
    proj = UtmProjector(frames[0].lat_deg, frames[0].lon_deg)

    out: list[UtmFrame] = []
    for fm in frames:
        e, n = proj.to_utm(fm.lat_deg, fm.lon_deg)
        out.append(UtmFrame(
            easting=e,
            northing=n,
            alt_m=fm.alt_m,
            east_rel=e - proj.east0,
            north_rel=n - proj.north0,
        ))
    return proj, out


def deltas_metrici(utm_frames: list[UtmFrame]) -> np.ndarray:
    """Delta(E, N) in metri tra scatti consecutivi. Shape (N-1, 2).

    Usata per ricavare la scala metrica del vettore traslazione VO senza
    dipendere dalla velocita' EXIF (rumorosa).
    """
    if len(utm_frames) < 2:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.array([(f.easting, f.northing) for f in utm_frames], dtype=np.float64)
    return np.diff(arr, axis=0)


def passi_metrici(utm_frames: list[UtmFrame]) -> np.ndarray:
    """|Delta posizione| (m) tra scatti consecutivi. Shape (N-1,)."""
    return np.linalg.norm(deltas_metrici(utm_frames), axis=1)


def seleziona_per_overlap(utm_frames: list[UtmFrame],
                          footprint_m: float,
                          target_overlap: float) -> list[int]:
    """Indici dei frame da tenere per ottenere `target_overlap` frontale.

    L'overlap frontale tra due scatti consecutivi vale (al primo ordine,
    nadir e moto lungo l'asse-immagine):

        overlap = 1 - gps_step / footprint_m

    quindi per un overlap target `p` lo spacing GPS ideale e' `f * (1 - p)`.
    Algoritmo greedy: parto da 0, accumulo la distanza GPS lungo il
    percorso e seleziono il frame quando supera `step_target_m`.

    Parameters
    ----------
    utm_frames : list[UtmFrame]
        Posizioni metriche (i passi vengono dalla loro distanza euclidea).
    footprint_m : float
        Dimensione tipica del frame al suolo lungo la direzione di volo
        (m). Tipicamente `H_img_px * GSD`.
    target_overlap : float
        Overlap target in [0, 1). Ad esempio 0.7 = 70%.

    Returns
    -------
    list[int]
        Indici nella lista in ingresso. Include sempre 0 e l'ultimo frame.
    """
    if not 0.0 <= target_overlap < 1.0:
        raise ValueError(f"target_overlap deve essere in [0, 1), ricevuto {target_overlap}")
    if footprint_m <= 0:
        raise ValueError(f"footprint_m deve essere > 0, ricevuto {footprint_m}")
    n = len(utm_frames)
    if n <= 1:
        return list(range(n))

    step_target = footprint_m * (1.0 - target_overlap)
    keep = [0]
    acc = 0.0
    last = utm_frames[0]
    for i in range(1, n):
        cur = utm_frames[i]
        acc += float(np.hypot(cur.easting - last.easting,
                              cur.northing - last.northing))
        last = cur
        if acc >= step_target:
            keep.append(i)
            acc = 0.0

    # Assicuro la presenza dell'ultimo frame: utile per chiudere il mosaico
    # anche quando il residuo di cammino e' minore di step_target.
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return keep
