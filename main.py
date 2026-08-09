"""Punto di ingresso.

    python main.py <volo> [--da N] [--a M] [--overlap-frontale F] [--overlap-laterale L]

Tutto cio' che caratterizza il volo -- range dei frame, soglie di virata, overlap,
risoluzione di lavoro -- viene dedotto dai dati gia' in nostro possesso (immagini su
disco, metadati EXIF/XMP, calibrazione). Le opzioni servono solo a scavalcare la
deduzione quando serve.

Le pose vengono stimate a risoluzione ridotta (veloce, e la localizzazione delle feature
non migliora abbastanza a piena risoluzione da giustificarne il costo), ma il mosaico
finale viene composto dalle immagini a piena risoluzione.
"""
import argparse
import sys
from pathlib import Path
import time

import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    blending,
    features,
    flight_filter,
    frames,
    georef,
    io_data,
    legs as legs_mod,
    matching,
    mosaic,
    odometry,
    overlap as overlap_mod,
    refine,
    resolution,
    transforms,
)
from utils.geodesy import gsd_meters_per_pixel, make_transformers


# Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CALIBRATION_FILE = DATA_DIR / "calibration.json"    # Prodotto da preprocessing/create_calibration.py
METADATA_FILE = DATA_DIR / "metadata.json"          # Prodotto da preprocessing/create_metadata.py
TRANSLATIONS_FILE = DATA_DIR / "translations.json"  # Prodotto da preprocessing/create_translations.py
OUTPUT_JPG = ROOT / "output" / "mosaic.jpg"
OUTPUT_TIFF = ROOT / "output" / "mosaic.tif"


# CLI
def _opt_int(raw: str) -> int | None:
    """Intero >= 1, oppure None se il valore e' vuoto o il letterale 'none'."""
    if raw.strip().lower() in ("", "none"):
        return None
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"atteso un intero o 'none', ricevuto {raw!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"i frame sono numerati da 1, ricevuto {value}")
    return value


def _opt_overlap(raw: str) -> float | None:
    """Frazione in [0, 1), oppure None se il valore e' vuoto o il letterale 'none'."""
    if raw.strip().lower() in ("", "none"):
        return None
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"atteso un numero o 'none', ricevuto {raw!r}")
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(
            f"overlap fuori dall'intervallo [0, 1): {value}"
        )
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Compone un mosaico georeferenziato dalle immagini di un volo drone. "
            "Senza opzioni processa l'intero volo, deducendo overlap e soglie di virata "
            "dai dati."
        ),
        epilog=(
            "esempi:\n"
            "  python main.py immagini/immagini_drone/immagini_senza_distorsione\n"
            "  python main.py <volo> --da 2 --a 89\n"
            "  python main.py <volo> --overlap-frontale 0.815 --overlap-laterale none"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "volo",
        type=Path,
        help="cartella con le immagini del volo (JPG non distorte, nomi ..._NNNN_D.JPG)",
    )
    parser.add_argument(
        "--da",
        type=_opt_int,
        default=None,
        metavar="N",
        help="primo frame dell'intervallo; se omesso parte dal primo frame presente",
    )
    parser.add_argument(
        "--a",
        type=_opt_int,
        default=None,
        metavar="M",
        help="ultimo frame dell'intervallo, incluso; se omesso arriva all'ultimo frame presente",
    )
    parser.add_argument(
        "--overlap-frontale",
        type=_opt_overlap,
        default=None,
        metavar="F",
        help="overlap frontale in [0, 1); se omesso usa il valore dedotto dal volo",
    )
    parser.add_argument(
        "--overlap-laterale",
        type=_opt_overlap,
        default=None,
        metavar="L",
        help="overlap laterale in [0, 1); se omesso usa il valore dedotto dal volo",
    )

    args = parser.parse_args(argv)
    if args.da is not None and args.a is not None and args.a < args.da:
        parser.error(f"--a ({args.a}) deve essere >= --da ({args.da})")
    return args


def _check_inputs(images_dir: Path) -> list[str]:
    missing: list[str] = []
    for path in (CALIBRATION_FILE, METADATA_FILE, TRANSLATIONS_FILE):
        if not path.is_file():
            missing.append(f"file mancante: {path}")
    if not images_dir.is_dir():
        missing.append(f"cartella mancante: {images_dir}")
    elif not any(p.suffix.lower() == ".jpg" for p in images_dir.iterdir()):
        missing.append(f"nessun .jpg in {images_dir}")
    return missing


# Funzione dell'algoritmo (TODO)
