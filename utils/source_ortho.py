"""Voli consegnati come una ortofoto per scatto, gia' georeferenziata.

Ogni file arriva ortorettificato e collocato per conto proprio (tag GDAL `SHOT_*`, CRS
geografico). `preprocessing.create_ortho_frames` ne ha gia' ricavato i fotogrammi
rettangolari a pixel quadrati e il loro `data/ortho_<volo>.json`; qui non resta che
distribuire quei numeri nei ruoli che la pipeline si aspetta.

Tre assenze e come vengono coperte:

- **niente calibrazione.** Non serve: la focale entra solo nel GSD, e il GSD questa
  consegna ce l'ha gia' scritto nel geotransform. `gsds` e' costante, dal ricampionamento.

- **niente bussola.** `SHOT_YAW` sembrerebbe l'imbardata ma non lo e' -- sul volo di prova
  vale -0,96 gradi su una passata e +2,18 sull'altra, mentre le rotte vere sono 143,5 e
  323,5. E' lo yaw residuo dell'ortofoto. La rotta vera la misura il pre-pass, dall'asse
  del rettangolo di ripresa: e' una misura sull'immagine, quindi vale molti decimali.

- **niente odometria.** Nel prodotto non ci sono velocita' inerziali, e non e' un problema
  di lettore: non sono state registrate. L'unica traslazione disponibile e' la posizione
  GNSS, quindi **qui il GPS semina la ricostruzione**, e il residuo del fit finale smette
  di essere una validazione indipendente. Da qui `gps_seeded=True`.

Restano pero' due misure distinte della stessa cosa, e conviene tenerle separate: il tag
`SHOT_LAT/LON` e il centro del fotogramma secondo il geotransform sono scostati di mezzo
metro sistematico -- 0,515 m di mediana e 0,849 m di massimo sul volo di prova, quasi
certamente il braccio di leva applicato da chi ha scritto l'ortofoto. Il tag semina il
frame locale, il geotransform fa da bersaglio al fit. Non e' indipendenza, perche' il
ricevitore e' lo stesso, ma tiene come riferimento la georeferenziazione consegnata e fa
uscire allo scoperto quello scarto invece di sommarlo in silenzio.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from preprocessing.create_ortho_frames import percorsi_per
from utils.geodesy import transformers_for
from utils.sources import Source


def missing_inputs(images_dir: Path) -> list[str]:
    frames_dir, json_path = percorsi_per(images_dir)
    if not json_path.is_file():
        return [
            f"file mancante: {json_path}",
            f"  generalo con: python -m preprocessing.create_ortho_frames {images_dir}",
        ]
    if not frames_dir.is_dir() or not any(frames_dir.glob("*.tif")):
        return [
            f"nessun fotogramma in {frames_dir}",
            f"  rigeneralo con: python -m preprocessing.create_ortho_frames {images_dir}",
        ]
    return []


def load(
    images_dir: Path, frame_start: int | None = None, frame_end: int | None = None
) -> Source:
    frames_dir, json_path = percorsi_per(images_dir)
    with open(json_path) as f:
        documento = json.load(f)

    gsd = float(documento["gsd"])
    w, h = documento["image_size"]
    utm_crs = documento["utm_crs"]
    to_utm, to_wgs84 = transformers_for(utm_crs)

    voci = sorted(documento["frames"], key=lambda v: v["index"])
    inizio = frame_start if frame_start is not None else voci[0]["index"]
    fine = frame_end if frame_end is not None else voci[-1]["index"]
    voci = [v for v in voci if inizio <= v["index"] <= fine]
    if not voci:
        raise ValueError(f"Nessun fotogramma nel range {inizio}..{fine}")

    mancanti = [v["frame"] for v in voci if not (frames_dir / v["frame"]).is_file()]
    if mancanti:
        raise ValueError(
            f"{len(mancanti)} fotogrammi dichiarati in {json_path.name} non sono su disco "
            f"(il primo e' {mancanti[0]}): rigenera il pre-pass."
        )

    # Stessa cautela di `dataset.verify_image_size`: le trasformazioni del json valgono per
    # fotogrammi di quella dimensione, e se i file su disco sono di un'altra il GSD, le
    # impronte e la rimappatura dei punti sono tutti sfasati dello stesso fattore, in
    # silenzio. Costa la lettura di un file.
    campione = cv2.imread(str(frames_dir / voci[0]["frame"]), cv2.IMREAD_COLOR)
    if campione is None:
        raise ValueError(f"Impossibile leggere {frames_dir / voci[0]['frame']}")
    if (campione.shape[1], campione.shape[0]) != (int(w), int(h)):
        raise ValueError(
            f"{json_path.name} dichiara fotogrammi {w}x{h}, ma {voci[0]['frame']} e' "
            f"{campione.shape[1]}x{campione.shape[0]}. Json e fotogrammi non vengono dalla "
            "stessa esecuzione: rigenera il pre-pass."
        )

    # `lat`/`lon` sono la georeferenziazione CONSEGNATA del centro fotogramma, non il tag
    # dello scatto: e' cio' contro cui il fit finale si chiude, e non deve entrare nel
    # frame locale.
    centri = np.array([v["center_utm"] for v in voci], dtype=np.float64)
    lon, lat = to_wgs84.transform(centri[:, 0], centri[:, 1])

    records = [
        {
            "index": v["index"],
            "path": frames_dir / v["frame"],
            "source": images_dir / v["source"],
            "lat": float(lat[k]),
            "lon": float(lon[k]),
            "flight_yaw_deg": float(v["yaw_deg"]),
            "frame_to_utm": np.asarray(v["tile_to_utm"], dtype=np.float64),
            "source_to_utm": np.asarray(v["source_to_utm"], dtype=np.float64),
            # `gimbal_pitch_deg` volutamente assente: queste immagini sono gia'
            # ortorettificate, quindi la premessa del criterio del nadir -- che l'impronta
            # sia il rettangolo solo se la camera guardava in giu' -- non si applica.
            # `flight.mark_curves` lo salta da solo quando la chiave non c'e'.
        }
        for k, v in enumerate(voci)
    ]

    # Il seed della traslazione: il tag dello scatto, in UTM, con origine sul primo.
    est, nord = to_utm.transform(
        [v["shot_lon"] for v in voci], [v["shot_lat"] for v in voci]
    )
    posizioni = np.column_stack([est, nord])
    posizioni -= posizioni[0]

    scarto = np.linalg.norm(centri - centri[0] - posizioni, axis=1)

    return Source(
        kind="ortho",
        source_dir=images_dir,
        records=records,
        positions_m=posizioni,
        gsds=np.full(len(records), gsd, dtype=np.float64),
        image_size=(int(w), int(h)),
        utm_crs=utm_crs,
        # Il frame locale nasce direttamente in UTM, quindi e' gia' in nord GRIGLIA: al fit
        # finale non resta nessuna convergenza del meridiano da assorbire.
        grid_north=True,
        gps_seeded=True,
        intro=[
            f"{len(records)} scatti (frame {records[0]['index']}..{records[-1]['index']}), "
            f"fotogrammi {w}x{h}, GSD {gsd * 1000:.3f} mm/px",
            f"ortofoto per scatto: le posizioni vengono dai tag SHOT_LAT/LON, quindi il "
            f"GPS semina la ricostruzione",
            f"tag e geotransform discordano di {np.median(scarto):.3f} m di mediana, "
            f"{scarto.max():.3f} m di massimo",
        ],
    )
