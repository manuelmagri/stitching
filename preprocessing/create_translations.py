import json
from pathlib import Path
from datetime import datetime

# Genera `data/translations.json`: i delta di traslazione inter-frame, in metri.
#
# Il file e' un array JSON con un elemento `[est, nord]` per frame, ottenuto
# integrando la velocita' di volo XMP sul tempo trascorso fra due scatti
# (EXIF:DateTimeOriginal). E' l'odometria che la pipeline usa come vincolo
# inter-frame (`utils.odometry.load_vo_deltas`).
#
# Attenzione agli assi: in XMP DJI `FlightXSpeed` e' la componente NORD e
# `FlightYSpeed` quella EST, non e' ENU standard. Lo scambio viene fatto qui, una
# volta sola, cosi' il file e' gia' nell'ordine (est, nord) usato dalla pipeline.
#
# Il primo elemento non ha un frame precedente da cui misurare il tempo: gli si
# assegna 1 s convenzionale, ed e' comunque trascurabile perche' al decollo il
# drone e' praticamente fermo.


# Path
ROOT = Path(__file__).resolve().parent.parent
METADATA_FILE_PATH = ROOT / "data" / "metadata.json"
OUTPUT_TRANSLATIONS_FILE_PATH = ROOT / "data" / "translations.json"


# Genera traslazioni
def genera_traslazioni(metadata_file, output_file):
    """Calcola i delta [est, nord] da `metadata_file` e li scrive in `output_file`.

    Ritorna la lista dei delta, uno per frame, nello stesso ordine dei metadati.
    """
    with open(metadata_file) as f:
        metadati = json.load(f)

    traslazioni = []
    istante_prec = None
    for entry in metadati:
        istante = datetime.strptime(str(entry["EXIF:DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S")
        delta_t = 1.0 if istante_prec is None else (istante - istante_prec).total_seconds()
        nord = float(entry["XMP:FlightXSpeed"])
        est = float(entry["XMP:FlightYSpeed"])
        traslazioni.append([est * delta_t, nord * delta_t])
        istante_prec = istante

    Path.mkdir(Path(output_file).parent, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(traslazioni, f, indent=2)

    print(f"Traslazioni salvate in {output_file} ({len(traslazioni)} frame)")
    return traslazioni


# Run
genera_traslazioni(METADATA_FILE_PATH, OUTPUT_TRANSLATIONS_FILE_PATH)