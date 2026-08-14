"""Intrinseci delle immagini RETTIFICATE, cioe' quelle che la pipeline consuma.

    python -m preprocessing.create_calibration immagini/<volo>

La pipeline non lavora sugli scatti originali ma su quelli prodotti da
`undistort_images`, che li corregge con una nuova matrice e li ritaglia al rettangolo
valido: focale, centro ottico e dimensione sono diversi da quelli scritti nell'XMP. Qui
va salvata la coppia RETTIFICATA, perche' la focale e' l'unico numero da cui dipende la
scala metrica (GSD = quota / focale).

Gli intrinseci sono identici per tutte le foto della stessa camera, quindi basta il
primo scatto della cartella. Arrivano da `_xmp.parametri_rettifica`, la stessa funzione
che usa `undistort_images`, cosi' calibrazione e immagini non possono divergere.

`imageSize` e' la dimensione attesa delle immagini rettificate, che la pipeline confronta
in avvio con i file su disco (`utils.dataset.verify_image_size`); `sorgente` conserva da
cosa e' stata derivata.
"""
import argparse
import json
from pathlib import Path

from preprocessing import CALIBRATION_FILE, scatti
from preprocessing._xmp import parametri_rettifica


def genera_calibrazione(cartella_volo: Path, output_file: Path):
    """Scrive gli intrinseci rettificati dedotti dal primo scatto ORIGINALE di
    `cartella_volo`. Ritorna la matrice 3x3 rettificata."""
    riferimento = scatti(cartella_volo)[0]
    parametri = parametri_rettifica(riferimento)
    K, K_orig = parametri["camera_matrix_rettificata"], parametri["camera_matrix"]
    w, h = parametri["image_size"]
    w_orig, h_orig = parametri["image_size_originale"]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(
            {
                "cameraMatrix": K.tolist(),
                "imageSize": [w, h],
                "sorgente": {
                    "immagine": str(riferimento),
                    "cameraMatrix": K_orig.tolist(),
                    "distCoeffs": parametri["dist_coeffs"].tolist(),
                    "imageSize": [w_orig, h_orig],
                    "roi": list(parametri["roi"]),
                },
            },
            f,
            indent=2,
        )

    print(f"Calibrazione da {riferimento.name} -> {output_file}")
    print(f"  originale   {w_orig}x{h_orig}  focale {K_orig[0, 0]:.2f} px")
    print(f"  rettificata {w}x{h}  focale {K[0, 0]:.2f} px  centro ({K[0, 2]:.1f}, {K[1, 2]:.1f})")
    return K


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrive data/calibration.json con gli intrinseci delle immagini rettificate.",
        epilog="esempio:\n  python -m preprocessing.create_calibration immagini/<volo>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cartella", type=Path, help="cartella degli scatti ORIGINALI")
    genera_calibrazione(parser.parse_args().cartella, CALIBRATION_FILE)
