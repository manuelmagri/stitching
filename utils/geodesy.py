"""Conversioni geodetiche: zona UTM, transformer WGS84<->UTM, GSD, convergenza del meridiano.

Questo modulo serve solo ai due estremi della pipeline -- la scelta del CRS e la
georeferenziazione finale. Lo stitching non lo usa: lavora nel frame locale metrico
costruito da `utils.localframe`, senza GPS.
"""
import math

import pyproj


def utm_epsg_for(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def central_meridian_for(epsg: int) -> float:
    """Longitudine del meridiano centrale della zona UTM identificata da `epsg`."""
    zone = epsg - (32600 if epsg < 32700 else 32700)
    return (zone - 1) * 6.0 - 180.0 + 3.0


def make_transformers(lat0: float, lon0: float):
    epsg = utm_epsg_for(lat0, lon0)
    utm_crs = f"EPSG:{epsg}"
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    return to_utm, to_wgs84, utm_crs


def gsd_meters_per_pixel(altitude_m: float, focal_px: float) -> float:
    """Ground sampling distance per camera nadir: m/px = altitudine / focale in pixel.

    `focal_px` deve essere la focale delle immagini RETTIFICATE (data/calibration.json),
    non quella dell'XMP: differiscono del 15% e questo e' l'unico numero da cui dipende
    tutta la scala metrica.
    """
    return altitude_m / focal_px


def meridian_convergence_deg(lat: float, lon: float) -> float:
    """Angolo fra nord vero e nord griglia UTM, in gradi, positivo verso est.

    La bussola del drone misura il nord VERO, UTM usa il nord GRIGLIA: chi costruisce
    un frame locale sull'assetto e poi lo scrive in UTM deve tenerne conto, altrimenti
    il mosaico esce ruotato. E' il valore atteso per la rotazione che `utils.georeference`
    assorbe nel fit finale, e serve come controllo di quel fit.
    """
    lon_c = central_meridian_for(utm_epsg_for(lat, lon))
    return math.degrees(
        math.atan(math.tan(math.radians(lon - lon_c)) * math.sin(math.radians(lat)))
    )
