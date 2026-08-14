"""Riporta le ortofoto per scatto al fotogramma che la pipeline si aspetta.

    python -m preprocessing.create_ortho_frames <cartella_volo>

Alcune camere non consegnano gli scatti grezzi ma una ortofoto per scatto: ogni file e'
gia' ortorettificato, ruotato a nord e georeferenziato per conto suo (tag GDAL `SHOT_*`,
CRS geografico). E' un formato di consegna diffuso -- lo producono anche Pix4D, Agisoft e
DJI Terra -- e ha due proprieta' che lo rendono inservibile cosi' com'e':

- **il pixel a terra non e' quadrato.** Il fotogramma nativo viene riscritto nel proprio
  bounding box nord-up mantenendo la dimensione raster, quindi `gsdy/gsdx` finisce per
  valere `aspect_raster / aspect_bbox`: 1,266 su un volo a 8 m, 1,338 su uno a 12 m.
  `utils.poses` parametrizza le pose come similarita', che hanno una sola scala: quello
  stiramento non avrebbe dove finire e sparirebbe in silenzio, deformando il mosaico.

- **meta' del riquadro e' vuota.** Il fotogramma sta dentro il suo bounding box ruotato,
  quindi i quattro angoli sono nodata e i pixel utili sono il 51%. Impronte, cuciture e
  feature ORB lavorerebbero tutte su un rettangolo che a terra non copre.

Qui il confezionamento viene disfatto invece di essere digerito a valle: si ricampiona
ogni ortofoto in un fotogramma RETTANGOLARE a pixel QUADRATI allineato alla ripresa. E'
lo stesso ruolo che `undistort_images` ha per gli scatti grezzi, e dopo questo passo
`footprint`, `features` e `compositing` non sanno nulla di ortofoto.

Che l'impronta valida sia davvero un rettangolo non e' un'ipotesi: riproiettandola in UTM
i suoi angoli interni stanno fra 88,0 e 91,6 gradi e i lati misurano 7,13 x 5,34 m a 8 m
di quota, cioe' il fotogramma nativo della camera. La deformazione visibile nei pixel del
file consegnato e' tutta e sola l'anisotropia.

Il fotogramma prodotto segue la convenzione DJI: lato lungo cross-track, lato corto
along-track, e il verso "su" dell'immagine (-y) lungo la rotta. Cosi' `yaw_deg` significa
la stessa cosa di `flight_yaw_deg` e `utils.poses.initial_pose` non distingue le due
sorgenti.

Scrive i fotogrammi in `<volo>_rettificate/` e `data/ortho_<volo>.json`; li consuma
`utils.source_ortho`. I conti geometrici stanno in `preprocessing._ortho`.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from preprocessing import DATA_DIR, _ortho
from utils.geodesy import make_transformers

# Quanto si rientra dal bordo dell'impronta valida prima di ritagliare il fotogramma.
# Non e' un margine di sicurezza generico: e' la frangia che il ricampionamento dello
# scrittore lascia contro il nodata, misurata sul dataset di prova come profilo di
# luminanza in funzione della distanza dal bordo -- mediana 2 a un pixel, 24 a due, 45 a
# quattro, contro 88 all'interno. Sotto i quattro pixel si porterebbe dentro al mosaico
# una cornice scura su ogni scatto.
MARGINE_PX = 4


def genera_fotogrammi(
    cartella_volo: Path,
    frames_dir: Path,
    output_json: Path,
    margine: int = MARGINE_PX,
) -> dict:
    """Scrive i fotogrammi rettangolari e il loro json. Ritorna il json come dict."""
    sorgenti = sorted(Path(cartella_volo).glob("*.tif"))
    if not sorgenti:
        raise FileNotFoundError(f"Nessun .tif in {cartella_volo}")

    # Il CRS proiettato si sceglie una volta per tutto il volo, dal primo scatto: e' il
    # sistema in cui i fotogrammi vengono misurati e in cui la pipeline lavorera'.
    primo = _ortho.leggi_tags(sorgenti[0])
    to_utm, to_wgs84, utm_crs = make_transformers(
        float(primo["SHOT_LAT"]), float(primo["SHOT_LON"])
    )

    print(f"{len(sorgenti)} ortofoto in {cartella_volo} | {utm_crs}")

    # Prima passata: geometria di ogni scatto. Il fotogramma comune si puo' decidere solo
    # dopo aver visto tutti, perche' `frames.FrameReader` assume una sola dimensione.
    letti = [_ortho.leggi_ortofoto(path, to_utm, margine) for path in sorgenti]
    rettangoli = [
        _ortho.footprint_rect_utm(info["mask_erosa"], info["px_to_utm"]) for info in letti
    ]

    centri = np.array([r.centro for r in rettangoli])
    corto_along = _ortho.lato_corto_e_along_track(centri, rettangoli)
    print(
        "  asse along-track: il lato "
        + ("CORTO" if corto_along else "LUNGO")
        + " del rettangolo di ripresa"
    )

    # GSD comune, quadrato: il piu' fine dei due assi su tutti gli scatti. Prendere il
    # piu' grossolano, o la media, butterebbe il dettaglio che l'asse fine ha davvero.
    gsd = min(min(i["gsd_xy"]) for i in letti)
    anisotropia = [max(i["gsd_xy"]) / min(i["gsd_xy"]) for i in letti]
    print(
        f"  anisotropia del pixel consegnato: da {min(anisotropia):.4f} a "
        f"{max(anisotropia):.4f}"
    )

    # Il primo tentativo di dimensione viene dal fotogramma PIU' PICCOLO del volo, non dal
    # piu' grande: cosi' ogni tile sta dentro la propria impronta valida e non resta un
    # solo pixel di nodata da mascherare a valle. Si perde l'1-2% ai bordi, che e' la parte
    # piu' obliqua del fotogramma e la meno affidabile.
    lato_lungo = min(r.lato_lungo for r in rettangoli)
    lato_corto = min(r.lato_corto for r in rettangoli)
    lato_cross, lato_along = (lato_lungo, lato_corto) if corto_along else (lato_corto, lato_lungo)
    image_size, validi = _ortho.dimensione_comune(
        letti, rettangoli, centri, corto_along, gsd, to_wgs84,
        base=(int(lato_cross / gsd), int(lato_along / gsd)),
    )
    print(
        f"  GSD comune {gsd * 1000:.3f} mm/px (quadrato) | fotogramma "
        f"{image_size[0]}x{image_size[1]} = {image_size[0] * gsd:.2f} x "
        f"{image_size[1] * gsd:.2f} m"
    )

    frames_dir.mkdir(parents=True, exist_ok=True)
    voci, residui, residui_sorgente = [], [], []
    centro_px = np.array([image_size[0] / 2.0, image_size[1] / 2.0, 1.0])
    for info, avanti, A, M, residuo in _ortho.geometrie(
        letti, rettangoli, centri, corto_along, gsd, image_size, to_wgs84
    ):
        residui.append(residuo)
        centro = (A @ centro_px)[:2]
        # I pixel del file CONSEGNATO non sono quelli del fotogramma: chi ha misurato
        # qualcosa sull'ortofoto originale -- per esempio i punti di un CSV -- ha bisogno
        # anche di questa, per ritrovarsi dopo il ricampionamento.
        S, residuo_sorgente = _ortho.affine_campionata(info["size"], info["px_to_utm"])
        residui_sorgente.append(residuo_sorgente)
        tile = _ortho.warp(info["immagine"], M, image_size, cv2.INTER_CUBIC)
        destinazione = frames_dir / info["path"].name
        cv2.imwrite(str(destinazione), tile, [cv2.IMWRITE_TIFF_COMPRESSION, 5])  # 5 = LZW

        tags = info["tags"]
        voci.append(
            {
                "index": _ortho.indice(info["path"].name),
                "source": info["path"].name,
                "frame": destinazione.name,
                # Il verso "su" del fotogramma, in gradi da nord in senso orario: e' la
                # stessa convenzione di XMP:FlightYawDegree.
                "yaw_deg": float(math.degrees(math.atan2(avanti[0], avanti[1]))),
                "shot_lat": float(tags["SHOT_LAT"]),
                "shot_lon": float(tags["SHOT_LON"]),
                "center_utm": [float(centro[0]), float(centro[1])],
                "tile_to_utm": A.tolist(),
                "source_to_utm": S.tolist(),
            }
        )

    voci.sort(key=lambda v: v["index"])
    documento = {
        "utm_crs": utm_crs,
        "gsd": gsd,
        "image_size": list(image_size),
        "margine_px": margine,
        "source_dir": str(Path(cartella_volo).resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "frames": voci,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(documento, f, indent=2)

    print(
        f"  residuo delle linearizzazioni UTM/lon-lat: {max(residui):.2e} px sul warp, "
        f"{max(residui_sorgente) * 1000:.2e} mm sul geotransform sorgente"
    )
    print(
        f"  copertura dell'impronta erosa: da {min(validi):.4%} a {max(validi):.4%} "
        f"(il riquadro consegnato ne copriva il 51%)"
    )
    print(f"  {len(voci)} fotogrammi in {frames_dir}")
    print(f"  {output_json}")
    return documento


def percorsi_per(cartella_volo: Path) -> tuple[Path, Path]:
    """(cartella dei fotogrammi, json) di un volo. Unico posto in cui si decidono.

    I fotogrammi stanno accanto alla consegna, in `<volo>_rettificate`, con la stessa
    convenzione di `undistort_images`: sono la stessa cosa -- la consegna portata sul
    contratto della pipeline -- e chi apre `immagini/` li trova di fianco all'originale
    invece che in un ramo di `data/`. Li' resta solo il json, che e' un file di scambio.
    """
    cartella = Path(cartella_volo).resolve()
    return (
        cartella.parent / f"{cartella.name}_rettificate",
        DATA_DIR / f"ortho_{cartella.name}.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Ricampiona le ortofoto per scatto in fotogrammi rettangolari a pixel "
            "quadrati, allineati alla rotta, che main.py consuma come un volo qualunque."
        ),
        epilog="esempio:\n  python -m preprocessing.create_ortho_frames immagini/georef",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cartella", type=Path, help="cartella con le ortofoto .tif")
    parser.add_argument(
        "--margine", type=int, default=MARGINE_PX, metavar="PX",
        help=(
            "pixel di rientro dal bordo dell'impronta valida, per lasciare fuori la "
            f"frangia scura del ricampionamento (default {MARGINE_PX})"
        ),
    )
    args = parser.parse_args()
    genera_fotogrammi(args.cartella, *percorsi_per(args.cartella), args.margine)
