"""Scrittura del mosaico come GeoTIFF in UTM, una banda alla volta.

Si scrive nel CRS proiettato in cui il mosaico e' gia' espresso, senza riproiettare in
lat/lon. Una ortofoto in UTM ha pixel quadrati in metri, si misura sopra direttamente, ed
e' la forma in cui questi prodotti normalmente circolano; chi ha bisogno di EPSG:4326 lo
ottiene con un gdalwarp sul file. La riproiezione in memoria, che la versione precedente
faceva banda per banda sull'array intero, su 1,27 gigapixel non sarebbe comunque stata
sostenibile: allocava sorgente e destinazione per tre volte.

L'alfa distingue i pixel effettivamente coperti da quelli lasciati vuoti ai bordi del
riquadro, che altrimenti sarebbero neri opachi.

Nel file si scrive anche da QUALI fotogrammi e' nato (vedi `input_stamp`). Serve perche' il
mosaico e i suoi ingressi hanno vite separate: il pre-pass si rifa', i fotogrammi cambiano
cartella, e un GeoTIFF vecchio resta li' indistinguibile da uno appena fatto. Senza timbro
la domanda "questo file e' aggiornato?" si risponde solo confrontando a occhio le
dimensioni del canvas, che e' come ci si accorge del guaio quando ormai si e' guardato per
ore un risultato che non c'entra.

La domanda ha senso perche' la pipeline e' riproducibile: tre esecuzioni di fila sullo
stesso volo danno lo stesso GeoTIFF byte per byte (108.236.997 byte, sha256 identico).
L'unico passo stocastico e' il RANSAC di `utils.matching`, e il seme dell'RNG di OpenCV non
e' fissato: la stabilita' e' misurata, non imposta. Se un giorno servisse imporla basta un
`cv2.setRNGSeed` all'avvio -- ma nasconderebbe la sensibilita' dei dati invece di mostrarla.
"""
import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window

# I tag finiscono nello spazio dei nomi principale del GeoTIFF, cosi' `gdalinfo` li mostra
# senza bisogno di sapere che esistono. Il prefisso evita collisioni con i tag di GDAL.
PREFISSO = "STITCHING_"


def _impronta(frames_dir: Path, suffix: str) -> tuple[str, int, str]:
    """Impronta di una cartella di fotogrammi: nome, dimensione e data di ogni file.

    Non legge i pixel. Su un volo da 835 scatti leggerli costerebbe decine di gigabyte per
    rispondere a una domanda che i metadati del filesystem gia' risolvono. Il prezzo e' che
    due pre-pass che producono file identici risultano diversi, perche' cambia la data --
    ed e' il verso giusto in cui sbagliare: quell'errore costa una ricomposizione, quello
    opposto costa guardare un mosaico vecchio credendolo nuovo.

    Ritorna (impronta, quanti file, data del piu' recente).
    """
    cartella = Path(frames_dir)
    if not cartella.is_dir():
        return "", 0, ""
    file = sorted(p for p in cartella.iterdir() if p.suffix.lower() == suffix.lower())
    digest = hashlib.sha256()
    recente = 0
    for p in file:
        stato = p.stat()
        digest.update(f"{p.name}|{stato.st_size}|{stato.st_mtime_ns}\n".encode())
        recente = max(recente, stato.st_mtime_ns)
    quando = (
        datetime.fromtimestamp(recente / 1e9).isoformat(sep=" ", timespec="seconds")
        if file
        else ""
    )
    return digest.hexdigest()[:16], len(file), quando


def input_stamp(source_dir: Path, frame_paths: list[Path], composed: str) -> dict[str, str]:
    """Il timbro da scrivere nel GeoTIFF: da dove vengono i pixel di questo mosaico.

    `frame_paths` sono i fotogrammi effettivamente composti; l'impronta pero' si calcola su
    TUTTA la cartella che li contiene, non solo su quelli. E' la cartella l'unita' che il
    pre-pass rigenera, ed e' quella che va confrontata per sapere se il mosaico e' vecchio:
    un'impronta ristretta al sottoinsieme cambierebbe anche solo cambiando `--da`/`--a`, e
    confonderebbe "ho chiesto un altro pezzo di volo" con "gli ingressi sono cambiati".
    """
    cartella = Path(frame_paths[0]).parent
    impronta, quanti, quando = _impronta(cartella, Path(frame_paths[0]).suffix)
    # Assoluti e canonici: altrimenti lo stesso volo indicato una volta come
    # `immagini/georef` e una come percorso assoluto darebbe due timbri diversi, e il
    # confronto direbbe "vecchio" per un mosaico che va benissimo.
    return {
        f"{PREFISSO}SOURCE_DIR": str(Path(source_dir).resolve()),
        f"{PREFISSO}FRAMES_DIR": str(cartella.resolve()),
        f"{PREFISSO}FRAMES_COUNT": str(quanti),
        f"{PREFISSO}FRAMES_MTIME": quando,
        f"{PREFISSO}FRAMES_DIGEST": impronta,
        f"{PREFISSO}COMPOSED": composed,
    }


def read_stamp(tiff_path: Path) -> dict[str, str]:
    """Il timbro di un mosaico gia' scritto. Vuoto se il file e' anteriore al timbro."""
    with rasterio.open(tiff_path) as ds:
        tag = ds.tags()
    return {k: v for k, v in tag.items() if k.startswith(PREFISSO)}


def stale_reasons(stamp: dict[str, str], current: dict[str, str]) -> list[str]:
    """In che cosa il mosaico non corrisponde piu' agli ingressi. Vuota se corrisponde.

    Confronta solo la parte che descrive gli INGRESSI: `COMPOSED` dice quale pezzo di volo
    era stato chiesto, non se i file sono cambiati, e metterlo qui farebbe dichiarare
    vecchio un mosaico solo perche' lo si e' fatto su un altro intervallo di frame.
    """
    if not stamp:
        return ["il mosaico non ha timbro: e' stato scritto prima che esistesse"]
    motivi = []
    for chiave, cosa in (
        ("SOURCE_DIR", "la cartella del volo"),
        ("FRAMES_DIR", "la cartella dei fotogrammi"),
        ("FRAMES_COUNT", "il numero di fotogrammi"),
        ("FRAMES_DIGEST", "il contenuto della cartella dei fotogrammi"),
    ):
        prima, adesso = stamp.get(PREFISSO + chiave, ""), current.get(PREFISSO + chiave, "")
        if prima != adesso:
            motivi.append(f"{cosa}: {prima or '(assente)'} -> {adesso or '(assente)'}")
    return motivi


def write_geotiff(
    output_path: Path,
    canvas_size: tuple[int, int],
    origin_m: tuple[float, float],
    gsd: float,
    utm_crs: str,
    bands,
    preview_max_side: int = 0,
    stamp: dict[str, str] | None = None,
) -> np.ndarray | None:
    """Scrive il mosaico consumando il generatore `bands`.

    `bands` produce (riga_iniziale, banda BGR uint8, maschera booleana), come
    `utils.compositing.compose_bands`. `origin_m` e' la coordinata UTM del pixel (0, 0).

    Con `preview_max_side > 0` costruisce anche una miniatura mentre le bande passano, e la
    restituisce: e' l'unico modo ragionevole di guardare il risultato, visto che un JPEG a
    piena risoluzione qui non e' un'opzione.

    `stamp` sono i tag che dicono da quali fotogrammi il mosaico e' nato (vedi
    `input_stamp`); si scrivono nello spazio dei nomi principale, quindi si leggono con un
    `gdalinfo` senza sapere che ci sono.

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
        if stamp:
            dst.update_tags(**stamp)

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


def _main(argv: list[str] | None = None) -> int:
    """`python -m utils.georef <mosaico.tif> [volo]` -- di che cosa e' fatto, ed e' vecchio?"""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m utils.georef",
        description=(
            "Mostra da quali fotogrammi e' nato un mosaico, e se corrisponde ancora a "
            "quelli che sono su disco adesso."
        ),
        epilog=(
            "esempi:\n"
            "  python -m utils.georef output/mosaic.tif\n"
            "  python -m utils.georef output/mosaic.tif immagini/georef"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mosaico", type=Path)
    parser.add_argument(
        "volo", type=Path, nargs="?", help="cartella del volo con cui confrontarlo"
    )
    args = parser.parse_args(argv)

    if not args.mosaico.is_file():
        print(f"file mancante: {args.mosaico}")
        return 1
    timbro = read_stamp(args.mosaico)
    if not timbro:
        print(f"{args.mosaico} non ha timbro: e' stato scritto prima che esistesse.")
        return 1
    for chiave in sorted(timbro):
        print(f"  {chiave[len(PREFISSO):]:<14} {timbro[chiave]}")
    if args.volo is None:
        return 0

    from utils import sources

    try:
        sorgente = sources.load(args.volo)
    except (ValueError, FileNotFoundError, OSError) as errore:
        print(f"\nnon riesco a leggere {args.volo}: {errore}")
        return 1
    adesso = input_stamp(sorgente.source_dir, [r["path"] for r in sorgente.records], "")
    motivi = stale_reasons(timbro, adesso)
    if not motivi:
        print(f"\n{args.mosaico.name} corrisponde ai fotogrammi che sono su disco adesso.")
        return 0
    print(f"\n{args.mosaico.name} NON corrisponde piu' agli ingressi:")
    for riga in motivi:
        print(f"  - {riga}")
    print(f"  rifallo con: python main.py {args.volo}")
    return 2


if __name__ == "__main__":
    import sys

    sys.exit(_main())
