import json
import argparse
from pathlib import Path

from preprocessing._xmp import parametri_rettifica


# Genera `data/calibration.json`: gli intrinseci delle immagini che la pipeline consuma.
#
# La pipeline non lavora sugli scatti originali ma su quelli prodotti da
# `undistort_image`, che li corregge con una nuova matrice e li ritaglia al rettangolo
# valido. Focale, centro ottico e dimensione di quelle immagini sono diversi da quelli
# scritti nell'XMP, quindi qui va salvata la coppia RETTIFICATA: la focale e' l'unico
# numero da cui dipende la scala metrica (GSD = quota / focale), e sbagliarla sfasa
# dello stesso fattore footprint, overlap, raggi di ricerca e scala del mosaico.
#
# I valori arrivano da `_xmp.parametri_rettifica`, la stessa funzione che usa
# `undistort_image`, cosi' calibrazione e immagini non possono divergere.
#
# `imageSize` e' la dimensione attesa delle immagini rettificate: serve alla pipeline
# per verificare in avvio che i file su disco corrispondano a questa calibrazione.
# `sorgente` conserva da cosa e' stata derivata, per poterla ricontrollare a colpo d'occhio.


# Path
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = ROOT / "data" / "calibration.json"


# CLI
def parse_args(argv : list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="create_calibration.py",
        description=(
            "Scrive data/calibration.json con gli intrinseci delle immagini rettificate, "
            "cioe' quelle che la pipeline consuma."
        ),
        epilog=(
            "Esempio:\n"
            " python -m preprocessing.create_calibration immagini/cartella_volo/DJI_0001_D.JPG"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "immagine",
        type=Path,
        help=(
            "Scatto ORIGINALE (non rettificato) di riferimento; gli intrinseci sono "
            "identici per tutte le foto della stessa camera"
        ),
    )

    args = parser.parse_args(argv)

    return args


# Genera calibrazione
def genera_calibrazione(immagine_riferimento, output_file):
    """
    Scrive in `output_file` gli intrinseci delle immagini rettificate, dedotti da
    `immagine_riferimento`, che deve essere uno scatto ORIGINALE: e' da quello che si
    ricavano sia gli intrinseci di partenza sia i parametri del ritaglio.
    Ritorna la matrice 3x3 rettificata.
    """
    parametri = parametri_rettifica(immagine_riferimento)
    K = parametri["camera_matrix_rettificata"]
    K_orig = parametri["camera_matrix"]
    w, h = parametri["image_size"]
    w_orig, h_orig = parametri["image_size_originale"]

    Path.mkdir(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "cameraMatrix": K.tolist(),
                "imageSize": [w, h],
                "sorgente": {
                    "immagine": str(immagine_riferimento),
                    "cameraMatrix": K_orig.tolist(),
                    "distCoeffs": parametri["dist_coeffs"].tolist(),
                    "imageSize": [w_orig, h_orig],
                    "roi": list(parametri["roi"]),
                },
            },
            f,
            indent=2,
        )

    print(f"Calibrazione salvata in {output_file}")
    print(f"  originale   {w_orig}x{h_orig}  focale {K_orig[0, 0]:.2f} px")
    print(f"  rettificata {w}x{h}  focale {K[0, 0]:.2f} px  centro ({K[0, 2]:.1f}, {K[1, 2]:.1f})")
    return K


# RUN
if __name__ == "__main__":
    args = parse_args()
    genera_calibrazione(args.immagine, OUTPUT_PATH)
