"""Odometria: delta di traslazione fra scatti consecutivi, in metri.

    python -m preprocessing.create_translations

Legge `data/metadata.json` e scrive `data/translations.json`, un elemento
`[est, nord]` per scatto, integrando la velocita' di volo XMP sul tempo trascorso fra
due scatti. Lo consuma `utils.localframe.load_vo_deltas`, che ci costruisce sopra il
seed delle posizioni: e' l'unica traslazione disponibile senza GPS.

Attenzione agli assi: in XMP DJI `FlightXSpeed` e' la componente NORD e `FlightYSpeed`
quella EST, non e' ENU standard. Lo scambio viene fatto qui, una volta sola, cosi' il
file e' gia' nell'ordine (est, nord) usato dalla pipeline.

Il primo elemento non ha uno scatto precedente da cui misurare il tempo: gli si assegna
1 s convenzionale, ed e' comunque irrilevante perche' `integrate_positions` lo scarta.
"""
import json
from datetime import datetime

from preprocessing import METADATA_FILE, TRANSLATIONS_FILE


def genera_traslazioni(metadata_file, output_file) -> list[list[float]]:
    """Calcola i delta [est, nord] da `metadata_file` e li scrive in `output_file`."""
    if not metadata_file.is_file():
        raise FileNotFoundError(
            f"{metadata_file} non esiste: lancia prima "
            "`python -m preprocessing.create_metadata immagini/<volo>`"
        )
    with open(metadata_file) as f:
        metadati = json.load(f)

    traslazioni, istante_prec = [], None
    for entry in metadati:
        istante = datetime.strptime(str(entry["EXIF:DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S")
        delta_t = 1.0 if istante_prec is None else (istante - istante_prec).total_seconds()
        nord = float(entry["XMP:FlightXSpeed"])
        est = float(entry["XMP:FlightYSpeed"])
        traslazioni.append([est * delta_t, nord * delta_t])
        istante_prec = istante

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(traslazioni, f, indent=2)

    print(f"Traslazioni di {len(traslazioni)} scatti -> {output_file}")
    return traslazioni


if __name__ == "__main__":
    genera_traslazioni(METADATA_FILE, TRANSLATIONS_FILE)
