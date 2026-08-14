"""Dai file consegnati agli input che la pipeline consuma.

Due catene, una per formato di consegna, che convergono sullo stesso contratto:
fotogrammi rettangolari, nadirali, a pixel quadrati, tutti della stessa dimensione.

    scatti DJI grezzi + XMP                    ortofoto per scatto
    ------------------------------------       -------------------
    create_calibration    intrinseci           create_ortho_frames
    undistort_images      pixel
    create_metadata       telemetria
    create_translations   odometria

La catena DJI va eseguita in quest'ordine, e i primi tre passi vogliono la cartella
degli scatti ORIGINALI: la rettifica non ricopia il blocco XMP, quindi dopo di essa
l'assetto non e' piu' leggibile. Puntare uno dei tre su una cartella gia' rettificata
fallisce subito, con un messaggio che dice questo.

I percorsi dei file di scambio si dichiarano qui una volta sola, e li importa sia chi
li scrive (questi script) sia chi li legge (`utils.source_exif`): finche' stanno in un
posto solo, produttore e consumatore non possono divergere.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EXIFTOOL = ROOT / "exiftool-13.53_64" / "exiftool.exe"

CALIBRATION_FILE = DATA_DIR / "calibration.json"
METADATA_FILE = DATA_DIR / "metadata.json"
TRANSLATIONS_FILE = DATA_DIR / "translations.json"


def scatti(cartella: Path) -> list[Path]:
    """Le .jpg di una cartella, in ordine di nome. Alza se non ce ne sono."""
    immagini = sorted(p for p in Path(cartella).iterdir() if p.suffix.lower() == ".jpg")
    if not immagini:
        raise FileNotFoundError(f"Nessuna .jpg in {cartella}")
    return immagini
