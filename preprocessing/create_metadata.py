import argparse
import json
from pathlib import Path
import subprocess

# Genera `data/metadati.json`: i metadati EXIF/XMP di ogni scatto del volo.
# Lo consumano `utils.io_data.load_metadata` (GPS, quota, assetto) e
# `preprocessing.create_translations` (velocita' di volo e istante dello scatto).


# Path
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE_PATH = ROOT / "data" / "metadata.json"
EXIFTOOL_PATH = ROOT / "exiftool-13.53_64" / "exiftool.exe"


# Tag exiftool
TAGS = [
    "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
    "-RelativeAltitude",
    "-GimbalYawDegree", "-GimbalPitchDegree", "-GimbalRollDegree",
    "-FlightYawDegree", "-FlightPitchDegree", "-FlightRollDegree",
    "-FlightXSpeed", "-FlightYSpeed",
    "-DateTimeOriginal",
]


# CLI
def parse_args(argv : list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="create_metadata.py",
        description=("Descrizione"),
        epilog=("Esempio:\n python preprocessing/create_metadata.py immagini/immagini_drone/cartella_volo"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "cartella",
        type=Path,
        help="Percorso della cartella delle immagini del volo",
    )

    args = parser.parse_args(argv)

    return args 


# Genera metadati
def genera_metadati(image_folder, output_file, exiftool_path, chunk_size=10):
    """
    Estrae i metadati da tutte le .jpg di `image_folder` e li scrive in `output_file`.
    Il risultato e' un array JSON, un oggetto per immagine. `chunk_size` controlla
    quante immagini exiftool processa per invocazione: serve a non superare il
    limite di lunghezza della command line di Windows. Ritorna la lista estratta.
    """
    # immagini = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    immagini = sorted(Path(image_folder).glob('*jpg'))
    if not immagini:
        raise FileNotFoundError(f"Nessuna .jpg in {image_folder}")

    metadati = []
    for i in range(0, len(immagini), chunk_size):
        result = subprocess.run(
            [exiftool_path, "-G", "-json", *TAGS, *immagini[i:i + chunk_size]],
            capture_output=True, text=True, check=True,
        )
        metadati.extend(json.loads(result.stdout))

    Path.mkdir(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metadati, f, indent=2)

    print(f"Metadati salvati in {output_file} ({len(metadati)} frame)")
    return metadati


# Run
args = parse_args()
genera_metadati(args.cartella, OUTPUT_FILE_PATH, EXIFTOOL_PATH)