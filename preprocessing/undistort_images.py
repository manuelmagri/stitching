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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
from tqdm import tqdm

from preprocessing import scatti
from preprocessing._xmp import parametri_rettifica

# Quanti scatti rettificare in parallelo. Una volta precalcolata la mappa, il tempo se ne
# va quasi tutto in decodifica e codifica JPEG, che cv2 esegue rilasciando il GIL: qui i
# thread lavorano davvero. Oltre gli otto non si guadagna piu' nulla -- il collo di
# bottiglia diventa il disco -- e ognuno tiene in memoria una coppia di fotogrammi.
WORKER = min(8, os.cpu_count() or 1)


def correggi_distorsione_cartella(input_dir: Path, output_dir: Path) -> list[Path]:
    """Rettifica tutte le .jpg di `input_dir` salvandole in `output_dir`, stesso nome."""
    immagini = scatti(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Gli intrinseci sono identici per tutte le foto della stessa camera: si deducono
    # dalla prima, una volta sola. E' la stessa funzione che usa `create_calibration`,
    # cosi' le immagini prodotte qui e la calibrazione scritta la' restano coerenti.
    parametri = parametri_rettifica(immagini[0])
    x, y, rw, rh = parametri["roi"]

    # Se gli intrinseci valgono per tutte, anche la mappa "pixel d'arrivo -> pixel di
    # partenza" che ne discende vale per tutte, e si costruisce una volta sola.
    # `cv2.undistort` invece la ricostruiva a ogni scatto, ed e' li' che se ne andava il
    # tempo: rifarla costa dieci volte piu' che usarla. In formato CV_16SC2 esce in due
    # pezzi -- `mappa` le coordinate sorgente in virgola fissa, `frazioni` la parte
    # sub-pixel con cui `remap` interpola -- ed e' il piu' rapido dei formati accettati.
    mappa, frazioni = cv2.initUndistortRectifyMap(
        parametri["camera_matrix"],
        parametri["dist_coeffs"],
        None,
        parametri["new_camera_matrix"],
        parametri["image_size_originale"],
        cv2.CV_16SC2,
    )
    # Ritagliandola subito alla ROI, `remap` produce gia' l'immagine finale: non si
    # rettificano i pixel dei bordi curvi, che il ritaglio butterebbe comunque via.
    mappa = mappa[y : y + rh, x : x + rw].copy()
    frazioni = frazioni[y : y + rh, x : x + rw].copy()

    def rettifica(src: Path) -> Path | None:
        """Legge, rettifica e riscrive uno scatto. None se il file non e' leggibile."""
        img = cv2.imread(str(src))
        if img is None:
            return None
        dst = output_dir / src.name
        cv2.imwrite(str(dst), cv2.remap(img, mappa, frazioni, cv2.INTER_LINEAR))
        return dst

    # `pool.map` restituisce gli esiti nell'ordine di `immagini`, non in quello in cui i
    # thread finiscono: l'elenco prodotto resta quello di sempre, e con esso il legame
    # posizionale fra scatto e risultato su cui si regge il conteggio delle saltate.
    with ThreadPoolExecutor(max_workers=WORKER) as pool:
        esiti = list(
            tqdm(pool.map(rettifica, immagini), total=len(immagini), desc="  rettifica")
        )

    prodotte = [dst for dst in esiti if dst is not None]
    saltate = [src.name for src, dst in zip(immagini, esiti) if dst is None]

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
