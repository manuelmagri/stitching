"""Voli consegnati come immagini rettificate piu' telemetria EXIF/XMP.

E' la consegna tipica di un drone DJI, e la forma per cui la pipeline e' nata: gli scatti
grezzi passano da `preprocessing.undistort_images`, la telemetria da
`preprocessing.create_metadata` e `create_translations`, gli intrinseci da
`create_calibration`.

Qui non c'e' logica nuova: e' il caricamento che stava dentro `main.carica_volo`, spostato
dietro il contratto di `utils.sources` perche' non sia piu' l'unico modo di arrivarci.

Quello che rende questa sorgente diversa dall'altra e' che il GPS non la tocca. Scala,
orientamento e forma vengono da barometro, bussola e odometria; `lat`/`lon` restano da
parte fino al fit finale, e per questo il residuo di quel fit misura davvero qualcosa.
"""
from pathlib import Path

import numpy as np

# I percorsi dei file di scambio li dichiara chi li scrive, e si importano da li': se
# stessero anche qui, produttore e consumatore potrebbero divergere in silenzio.
from preprocessing import CALIBRATION_FILE, METADATA_FILE, TRANSLATIONS_FILE
from utils import dataset, localframe
from utils.geodesy import make_transformers
from utils.sources import Source


def missing_inputs(images_dir: Path) -> list[str]:
    mancanti: list[str] = []
    for path in (CALIBRATION_FILE, METADATA_FILE, TRANSLATIONS_FILE):
        if not path.is_file():
            mancanti.append(f"file mancante: {path}  (generalo con preprocessing/)")
    if not any(p.suffix.lower() == ".jpg" for p in images_dir.iterdir()):
        mancanti.append(f"nessun .jpg in {images_dir}")
    return mancanti


def load(
    images_dir: Path, frame_start: int | None = None, frame_end: int | None = None
) -> Source:
    calibrazione = dataset.load_calibration(CALIBRATION_FILE)
    records = dataset.load_records(METADATA_FILE)
    records = dataset.select_range(
        records, dataset.list_image_paths(images_dir), frame_start, frame_end
    )
    # La focale descrive le immagini RETTIFICATE: se i file su disco non sono quelli, ogni
    # misura metrica a valle e' sfasata dello stesso fattore, in silenzio.
    dataset.verify_image_size(calibrazione, records[0]["path"])

    posizioni = localframe.positions_for(
        records,
        localframe.integrate_positions(localframe.load_vo_deltas(TRANSLATIONS_FILE)),
    )
    gsds = localframe.frame_gsds(records, calibrazione.focal_px)

    lat = float(np.mean([r["lat"] for r in records]))
    lon = float(np.mean([r["lon"] for r in records]))

    return Source(
        kind="exif",
        source_dir=images_dir,
        records=records,
        positions_m=posizioni,
        gsds=gsds,
        image_size=calibrazione.image_size,
        utm_crs=make_transformers(lat, lon)[2],
        # La bussola misura il nord VERO: la convergenza del meridiano resta da assorbire
        # al fit finale, ed e' la' che va cercata come controllo.
        grid_north=False,
        gps_seeded=False,
        intro=[
            f"{len(records)} scatti (frame {records[0]['index']}..{records[-1]['index']}), "
            f"immagini {calibrazione.image_size[0]}x{calibrazione.image_size[1]}, "
            f"focale {calibrazione.focal_px:.2f} px"
        ],
    )
