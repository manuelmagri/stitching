"""Scrittura del mosaico come GeoTIFF in EPSG:4326 (WGS84 lat/lon).

Strategia: il mosaico esiste in coordinate metriche UTM (origine + GSD). Si scrive prima in UTM
come trasformata in memoria, poi rasterio.warp.reproject lo proietta in EPSG:4326. Se viene
fornita una maschera alpha, l'output ha 4 bande (RGBA) e i pixel fuori dall'area mosaicata
sono trasparenti invece che neri.
"""
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject


def write_geotiff(
    mosaic_bgr: np.ndarray,
    canvas_offset_px: tuple[int, int],
    ref_origin_m: tuple[float, float],
    gsd: float,
    utm_crs: str,
    output_path: Path,
    alpha_mask: np.ndarray | None = None,
) -> None:
    """mosaic_bgr: HxWx3 BGR uint8.
    alpha_mask: HxW bool/uint8, True/255 dove c'e' contenuto. Se None il TIFF e' RGB a 3 bande.
    ref_origin_m: (east, north) in UTM del pixel (0,0).
    gsd: m/px.
    """
    height, width = mosaic_bgr.shape[:2]
    east0, north0 = ref_origin_m
    transform_utm = from_origin(east0, north0, gsd, gsd)
    src_crs = CRS.from_string(utm_crs)
    dst_crs = CRS.from_epsg(4326)

    rgb = mosaic_bgr[..., ::-1]  # BGR -> RGB
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, dst_crs, width, height, *_bounds(transform_utm, width, height)
    )

    use_alpha = alpha_mask is not None
    n_bands = 4 if use_alpha else 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": dst_h,
        "width": dst_w,
        "count": n_bands,
        "dtype": "uint8",
        "crs": dst_crs,
        "transform": dst_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    if use_alpha:
        profile["photometric"] = "rgb"

    with rasterio.open(output_path, "w", **profile) as dst:
        # Imposta colorinterp PRIMA di scrivere le bande perche' GDAL associ correttamente
        # il tag EXTRA_SAMPLES alla banda alpha nel TIFF.
        if use_alpha:
            dst.colorinterp = (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )

        for band in range(3):
            src_band = np.ascontiguousarray(rgb[:, :, band])
            dst_band = np.zeros((dst_h, dst_w), dtype=np.uint8)
            reproject(
                source=src_band,
                destination=dst_band,
                src_transform=transform_utm,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            dst.write(dst_band, band + 1)

        if use_alpha:
            alpha_src = (np.asarray(alpha_mask).astype(bool).astype(np.uint8) * 255)
            dst_alpha = np.zeros((dst_h, dst_w), dtype=np.uint8)
            reproject(
                source=np.ascontiguousarray(alpha_src),
                destination=dst_alpha,
                src_transform=transform_utm,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )
            dst.write(dst_alpha, 4)


def _bounds(transform, width: int, height: int):
    left = transform.c
    top = transform.f
    right = left + transform.a * width
    bottom = top + transform.e * height
    return left, bottom, right, top
