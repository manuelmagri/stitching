"""Rettifica gli scatti di un volo DJI: corregge la distorsione e ritaglia al valido.

    python -m preprocessing.undistort_images immagini/<volo> [--output DIR]

E' cio' che porta la consegna DJI sul contratto della pipeline -- fotogrammi
rettangolari, nadirali, a pixel quadrati -- lo stesso ruolo che `create_ortho_frames`
ha per le ortofoto per scatto.

Il nome del file viene mantenuto: e' l'unico legame fra un'immagine rettificata e la sua
telemetria, perche' `utils.dataset` ricava il numero di frame dal nome. L'XMP invece non
sopravvive alla riscrittura, ed e' il motivo per cui `create_metadata` va lanciato sulla
cartella ORIGINALE e non su questa.
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from preprocessing import scatti
from preprocessing._xmp import parametri_rettifica


def correggi_distorsione_cartella(input_dir: Path, output_dir: Path) -> list[Path]:
    """Rettifica tutte le .jpg di `input_dir` salvandole in `output_dir`, stesso nome."""
    immagini = scatti(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gli intrinseci sono identici per tutte le foto della stessa camera: si deducono
    # dalla prima, una volta sola. E' la stessa funzione che usa `create_calibration`,
    # cosi' le immagini prodotte qui e la calibrazione scritta la' restano coerenti.
    parametri = parametri_rettifica(immagini[0])
    x, y, rw, rh = parametri["roi"]

    prodotte, saltate = [], []
    for src in tqdm(immagini, desc="  rettifica"):
        img = cv2.imread(str(src))
        if img is None:
            saltate.append(src.name)
            continue
        rettificata = cv2.undistort(
            img,
            parametri["camera_matrix"],
            parametri["dist_coeffs"],
            None,
            parametri["new_camera_matrix"],
        )
        dst = output_dir / src.name
        cv2.imwrite(str(dst), rettificata[y : y + rh, x : x + rw])
        prodotte.append(dst)

    print(f"{len(prodotte)} immagini {rw}x{rh} in {output_dir}")
    if saltate:
        print(f"  {len(saltate)} saltate, lettura fallita (la prima e' {saltate[0]})")
    return prodotte


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Corregge la distorsione degli scatti di un volo e li ritaglia al valido.",
        epilog="esempio:\n  python -m preprocessing.undistort_images immagini/<volo>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cartella", type=Path, help="cartella degli scatti ORIGINALI")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="DIR",
        help="dove scrivere le rettificate (default: <cartella>_rettificate, accanto all'originale)",
    )
    args = parser.parse_args()
    destinazione = args.output or args.cartella.parent / f"{args.cartella.name}_rettificate"
    correggi_distorsione_cartella(args.cartella, destinazione)
