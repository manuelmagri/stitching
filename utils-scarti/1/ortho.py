"""Scrittura del mosaico georeferenziato.

Niente nuove dipendenze pesanti (rasterio/GDAL): produciamo:
    * `mosaic.png`      l'immagine pixel
    * `mosaic.pgw`      worldfile ESRI con la affine pixel -> UTM
    * `mosaic.prj`      WKT del sistema di riferimento (UTM/WGS84)

Il worldfile ha 6 righe (in metri):
    A   X-pixel size (m/px lungo riga)        =  +g
    D   rotation row -> X (in genere 0)        =   0
    B   rotation col -> Y (in genere 0)        =   0
    E   Y-pixel size (negativo: y cresce in giu`) = -g
    C   X-coord del centro del pixel (0, 0)
    F   Y-coord del centro del pixel (0, 0)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .compose import GlobalMosaic


_WKT_UTM_TEMPLATE = (
    'PROJCS["WGS 84 / UTM zone {zone}{hemi}",'
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],'
    'PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",{cm}],'
    'PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],'
    'PARAMETER["false_northing",{fn}],'
    'UNIT["metre",1]]'
)


def _utm_wkt(epsg: int) -> str:
    if 32601 <= epsg <= 32660:
        zone = epsg - 32600
        hemi = "N"
        fn = 0
    elif 32701 <= epsg <= 32760:
        zone = epsg - 32700
        hemi = "S"
        fn = 10000000
    else:
        # Fallback a WGS84 generico (non dovrebbe accadere)
        return f'PROJCS["EPSG:{epsg}"]'
    cm = -177 + (zone - 1) * 6
    return _WKT_UTM_TEMPLATE.format(zone=zone, hemi=hemi, cm=cm, fn=fn)


def write_georeferenced_mosaic(
    mosaic: GlobalMosaic,
    out_path: Path,
    epsg_utm: int,
) -> dict:
    """Salva PNG + PGW (worldfile) + PRJ. Ritorna i path scritti."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_path = out_path.with_suffix(".png")
    wld_path = out_path.with_suffix(".pgw")
    prj_path = out_path.with_suffix(".prj")

    cv2.imwrite(str(img_path), mosaic.image)

    g = float(mosaic.gsd_m_per_px)
    # PGW: A,D,B,E,C,F (definizione ESRI, riferimento al CENTRO del pixel (0,0))
    A = g
    D = 0.0
    B = 0.0
    E = -g
    C = mosaic.min_E + g / 2.0
    F = mosaic.max_N - g / 2.0
    with open(wld_path, "w") as f:
        f.write(f"{A:.10f}\n{D:.10f}\n{B:.10f}\n{E:.10f}\n{C:.10f}\n{F:.10f}\n")

    with open(prj_path, "w") as f:
        f.write(_utm_wkt(epsg_utm))

    return {"image": img_path, "world": wld_path, "prj": prj_path}
