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
