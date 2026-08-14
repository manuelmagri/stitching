"""Telemetria EXIF/XMP di ogni scatto del volo.

    python -m preprocessing.create_metadata immagini/<volo>

Va lanciato sulla cartella ORIGINALE: la rettifica non ricopia il blocco XMP, quindi
sulle immagini prodotte da `undistort_images` l'assetto non c'e' piu'.

Scrive `data/metadata.json`; lo leggono `utils.dataset.load_records` (GPS, quota,
assetto) e `preprocessing.create_translations` (velocita' di volo e istante di scatto).

I tag estratti sono esattamente quelli che qualcuno legge, e non uno di piu': gimbal yaw
e roll, flight pitch e roll erano finiti nel file e nei record senza avere un lettore --
l'assetto del velivolo non dice nulla sull'immagine, perche' il gimbal e' stabilizzato
(vedi `utils.flight`). Aggiungerne uno qui vuol dire aggiungere anche chi lo consuma.
"""
import argparse
import json
import subprocess
from pathlib import Path

from preprocessing import EXIFTOOL, METADATA_FILE, scatti

TAGS = [
    "-GPSLatitude", "-GPSLongitude",                  # posizione: bersaglio del fit finale
    "-RelativeAltitude",                              # quota barometrica: la scala
    "-GimbalPitchDegree",                             # scarto dal nadir: scatti in virata
    "-FlightYawDegree",                               # bussola: l'orientamento
    "-FlightXSpeed", "-FlightYSpeed",                 # velocita': l'odometria
    "-DateTimeOriginal",                              # tempo su cui integrarla
]


def genera_metadati(
    image_folder: Path, output_file: Path, exiftool_path: Path, chunk_size: int = 10
) -> list[dict]:
    """Estrae i tag da tutte le .jpg di `image_folder` e li scrive in `output_file`.

    Il risultato e' un array JSON, un oggetto per immagine. `chunk_size` e' quante
    immagini exiftool processa per invocazione: serve a non superare il limite di
    lunghezza della command line di Windows.
    """
    immagini = scatti(image_folder)
    if not Path(exiftool_path).is_file():
        raise FileNotFoundError(f"exiftool non trovato in {exiftool_path}")

    metadati: list[dict] = []
    for i in range(0, len(immagini), chunk_size):
        blocco = [str(p) for p in immagini[i : i + chunk_size]]
        esito = subprocess.run(
            [str(exiftool_path), "-G", "-json", *TAGS, *blocco],
            capture_output=True, text=True, check=True,
        )
        metadati.extend(json.loads(esito.stdout))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metadati, f, indent=2)

    print(f"Metadati di {len(metadati)} scatti -> {output_file}")
    return metadati


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrive data/metadata.json con la telemetria EXIF/XMP del volo.",
        epilog="esempio:\n  python -m preprocessing.create_metadata immagini/<volo>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cartella", type=Path, help="cartella degli scatti ORIGINALI")
    genera_metadati(parser.parse_args().cartella, METADATA_FILE, EXIFTOOL)
