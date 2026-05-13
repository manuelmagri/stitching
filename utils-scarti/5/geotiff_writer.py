"""Scrittura del mosaico come GeoTIFF UTM apribile in QGIS.

Usa rasterio per applicare il CRS (EPSG UTM scelto da gps_utm.UtmProjector)
e la trasformazione affine pixel <-> coordinate UTM assolute. Le coordinate
UTM-assolute sono ottenute aggiungendo l'origine UTM (east0, north0) salvata
dal proiettore alle coordinate UTM-locali del canvas.
"""

from __future__ import annotations

from __future__ import annotations

import os

import numpy as np

from .gps_utm import UtmProjector
from .mosaic import MosaicCanvas

try:
    import rasterio
    from rasterio.enums import ColorInterp
    from rasterio.transform import Affine
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "rasterio non installato. Aggiungi `rasterio` a requirements.txt e installa."
    ) from exc


def scrivi_geotiff(canvas: MosaicCanvas,
                   projector: UtmProjector,
                   output_path: str,
                   alpha_mask: np.ndarray | None = None) -> str:
    """Salva `canvas.image` come GeoTIFF UTM (EPSG di `projector`) con banda alpha.

    Conversione canale: l'immagine OpenCV e' BGR; rasterio si aspetta in
    genere RGB. Riordino i canali in scrittura.

    Parameters
    ----------
    alpha_mask : np.ndarray | None
        Maschera booleana (H, W): True = pixel coperto da almeno un frame,
        False = vuoto. Se None la calcolo dai pixel non-zero del canvas.
        I pixel a False ricevono alpha=0 (trasparenti in QGIS).
    """
    img = canvas.image
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Atteso BGR (H,W,3), ricevuto shape {img.shape}")

    if alpha_mask is None:
        alpha_mask = img.any(axis=2)
    elif alpha_mask.shape != img.shape[:2]:
        raise ValueError(
            f"alpha_mask {alpha_mask.shape} non combacia con canvas {img.shape[:2]}"
        )

    # Pixel (0, 0) del canvas corrisponde a:
    #   Xw_locale = bounds[0]   (xmin del bounding box mondo locale)
    #   Yw_locale = bounds[3]   (ymax: l'asse Y mondo va in alto)
    # In assoluto sommiamo l'origine UTM del proiettore.
    xmin_loc, _ymin_loc, _xmax_loc, ymax_loc = canvas.bounds_world
    east0 = projector.east0 + xmin_loc
    north0 = projector.north0 + ymax_loc

    res = canvas.res_m_per_px
    # Affine: row cresce verso il basso, quindi `e` e' negativo.
    transform = Affine(res, 0.0, east0, 0.0, -res, north0)

    rgb = img[:, :, ::-1]  # BGR -> RGB
    alpha = (alpha_mask.astype(np.uint8) * 255)
    height, width = rgb.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    profile = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=4,
        dtype=rgb.dtype,
        crs=projector.crs.to_wkt(),
        transform=transform,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)
        dst.write(alpha, 4)
        # Marco esplicitamente la quarta banda come alpha: QGIS la usa per
        # rendere trasparenti i pixel a 0 senza configurazione aggiuntiva.
        dst.colorinterp = (
            ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha,
        )

    return output_path
