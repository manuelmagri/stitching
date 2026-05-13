"""Punto di ingresso. Modifica le costanti in CONFIG e lancia `python main.py`."""
import sys
from pathlib import Path

from utils import run_pipeline

ROOT = Path(__file__).resolve().parent

CONFIG = {
    # --- Percorsi ---
    "calibration_file": ROOT / "data" / "calibration.txt",
    "metadata_file": ROOT / "data" / "metadati.txt",
    "images_dir": ROOT / "immagini" / "immagini_drone" / "immagini_senza_distorsione",
    "output_jpg": ROOT / "output" / "mosaic.jpg",
    "output_tiff": ROOT / "output" / "mosaic.tif",

    # --- Range frame (1-indexed, estremi inclusi). frame_end=None per "fino alla fine". ---
    "frame_start": 2,
    "frame_end": 807,

    # --- Parametri di volo (per ora non usati, riservati per la prossima iterazione) ---
    "lateral_overlap": 0.81,
    "frontal_overlap": 0.81,

    # --- Performance ---
    "downscale": 2.5,

    # --- Filtro frame in curva ---
    "skip_curves": True,
    "roll_threshold_deg": 10.0,
    "yaw_rate_threshold_deg": 5.0,
}


def _check_inputs(cfg: dict) -> list[str]:
    missing: list[str] = []
    for key in ("calibration_file", "metadata_file"):
        if not cfg[key].is_file():
            missing.append(f"file mancante: {cfg[key]}")
    images_dir: Path = cfg["images_dir"]
    if not images_dir.is_dir():
        missing.append(f"cartella mancante: {images_dir}")
    else:
        if not any(p.suffix.lower() == ".jpg" for p in images_dir.iterdir()):
            missing.append(f"nessun .jpg in {images_dir}")
    return missing


def _validate_range(cfg: dict) -> str | None:
    start, end = cfg["frame_start"], cfg["frame_end"]
    if not isinstance(start, int) or start < 1:
        return f"frame_start non valido: {start}"
    if end is not None and (not isinstance(end, int) or end < start):
        return f"frame_end non valido (deve essere >= {start}): {end}"
    return None


if __name__ == "__main__":
    missing = _check_inputs(CONFIG)
    range_err = _validate_range(CONFIG)
    if missing or range_err:
        print("Errore di configurazione:")
        for m in missing:
            print(f"  - {m}")
        if range_err:
            print(f"  - {range_err}")
        sys.exit(1)
    run_pipeline(CONFIG)
