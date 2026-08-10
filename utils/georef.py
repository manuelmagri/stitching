"""Scrittura del mosaico come GeoTIFF in UTM, una banda alla volta.

Si scrive nel CRS proiettato in cui il mosaico e' gia' espresso, senza riproiettare in
lat/lon. Una ortofoto in UTM ha pixel quadrati in metri, si misura sopra direttamente, ed
e' la forma in cui questi prodotti normalmente circolano; chi ha bisogno di EPSG:4326 lo
ottiene con un gdalwarp sul file. La riproiezione in memoria, che la versione precedente
faceva banda per banda sull'array intero, su 1,27 gigapixel non sarebbe comunque stata
sostenibile: allocava sorgente e destinazione per tre volte.

L'alfa distingue i pixel effettivamente coperti da quelli lasciati vuoti ai bordi del
riquadro, che altrimenti sarebbero neri opachi.
"""
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window


def write_geotiff(
    output_path: Path,
    canvas_size: tuple[int, int],
    origin_m: tuple[float, float],
    gsd: float,
    utm_crs: str,
    bands,
    build_overviews: bool = True,
    preview_max_side: int = 0,
) -> np.ndarray | None:
    """Scrive il mosaico consumando il generatore `bands`.

    `bands` produce (riga_iniziale, banda BGR uint8, maschera booleana), come
    `utils.compositing.compose_bands`. `origin_m` e' la coordinata UTM del pixel (0, 0).

    Con `preview_max_side > 0` costruisce anche una miniatura mentre le bande passano, e la
    restituisce: e' l'unico modo ragionevole di guardare il risultato, visto che un JPEG a
    piena risoluzione qui non e' un'opzione.

    Le piramidi interne servono a chi apre il file: senza, un GIS deve leggere tutti i
    pixel per disegnare una vista d'insieme.
    """
    canvas_w, canvas_h = canvas_size
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fattore = min(1.0, preview_max_side / max(canvas_w, canvas_h)) if preview_max_side else 0.0
    anteprima = (
        np.zeros((max(int(canvas_h * fattore), 1), max(int(canvas_w * fattore), 1), 3), np.uint8)
        if fattore
        else None
    )

    profilo = {
        "driver": "GTiff",
        "width": canvas_w,
        "height": canvas_h,
        "count": 4,
        "dtype": "uint8",
        "crs": CRS.from_string(utm_crs),
        "transform": from_origin(origin_m[0], origin_m[1], gsd, gsd),
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "photometric": "rgb",
        "BIGTIFF": "IF_SAFER",
    }

    with rasterio.open(output_path, "w", **profilo) as dst:
        # Va impostato PRIMA di scrivere, perche' GDAL associ il tag EXTRA_SAMPLES
        # alla quarta banda invece di trattarla come un canale generico.
        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )

        for riga, banda, maschera in bands:
            altezza = banda.shape[0]
            finestra = Window(0, riga, canvas_w, altezza)
            rgb = banda[..., ::-1]  # BGR -> RGB
            for k in range(3):
                dst.write(np.ascontiguousarray(rgb[:, :, k]), k + 1, window=finestra)
            dst.write(maschera.astype(np.uint8) * 255, 4, window=finestra)

            if anteprima is not None:
                # Entrambi gli estremi vanno calcolati dalle righe ASSOLUTE del canvas.
                # Ricavare la fine da `y0 + altezza*fattore` fa perdere una riga ogni volta
                # che i due arrotondamenti cadono da parti opposte, e nell'anteprima
                # comparivano righe nere ai confini fra bande che nel GeoTIFF non c'erano.
                y0 = int(riga * fattore)
                y1 = min(int((riga + altezza) * fattore), anteprima.shape[0])
                if y1 > y0:
                    anteprima[y0:y1] = cv2.resize(
                        banda, (anteprima.shape[1], y1 - y0), interpolation=cv2.INTER_AREA
                    )

        if build_overviews:
            dst.build_overviews([2, 4, 8, 16, 32], Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    return anteprima


def canvas_origin_utm(
    reference_origin_m: tuple[float, float],
    canvas_offset_px: tuple[int, int],
    gsd: float,
) -> tuple[float, float]:
    """Coordinata UTM del pixel (0, 0) del canvas finale.

    `compute_canvas` trasla le pose perche' il riquadro parta dall'origine; l'offset che
    restituisce dice di quanto, e va riportato sulle coordinate del mondo. La y del canvas
    cresce verso sud, quindi il nord va sottratto.
    """
    off_x, off_y = canvas_offset_px
    return (
        reference_origin_m[0] + off_x * gsd,
        reference_origin_m[1] - off_y * gsd,
    )
