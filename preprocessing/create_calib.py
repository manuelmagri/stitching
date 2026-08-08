import json
import argparse
from pathlib import Path
from preprocessing._xmp import leggi_intrinseci


# Genera `data/calibration.json`: la matrice intrinseca della camera.
# E' l'unico dato di calibrazione che la pipeline consuma 
# (`utils.io_data.load_camera_matrix`, che ne legge la chiave "cameraMatrix").


# Path
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "calibration.json"


# CLI
def parse_args(argv : list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="create_calib.py",
        description=("Descrizione"),
        epilog=("Esempio:\n python create_calib.py immagini/immagini_drone/cartella_volo"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "immagine",
        type=Path,
        help="Percorso dell'immagine di riferimento",
    )

    args = parser.parse_args(argv)

    return args 


# Genera calibrazione
def genera_calibrazione(immagine_riferimento, output_file):
    """
    Scrive in `output_file` la camera matrix letta dall'XMP di `immagine_riferimento`.
    Gli intrinseci sono identici per tutte le foto della stessa camera, quindi
    una sola immagine basta a calibrare l'intero volo. Ritorna la matrice 3x3.
    """
    
    camera_matrix, _ = leggi_intrinseci(immagine_riferimento)

    Path.mkdir(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"cameraMatrix": camera_matrix.tolist()}, f, indent=2)

    print(f"Calibrazione salvata in {output_file}")
    return camera_matrix


# RUN
args = parse_args()
genera_calibrazione(args.immagine, OUTPUT_PATH)