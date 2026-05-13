"""Conversioni geodetiche: scelta zona UTM, transformer WGS84<->UTM, ground sample distance."""
import math

import pyproj


def utm_epsg_for(lat: float, lon: float) -> int:
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    return (32600 if lat >= 0 else 32700) + zone


def make_transformers(lat0: float, lon0: float):
    epsg = utm_epsg_for(lat0, lon0)
    utm_crs = f"EPSG:{epsg}"
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    return to_utm, to_wgs84, utm_crs


def gsd_meters_per_pixel(altitude_m: float, focal_px: float) -> float:
    """Ground sampling distance per camera nadir: m/px = altitudine / focale in pixel."""
    return altitude_m / focal_px
