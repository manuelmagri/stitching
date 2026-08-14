"""Punti misurati sui singoli scatti, riportati sul mosaico.

Accanto alle immagini di un volo possono arrivare dei CSV di punti -- rilievi, bersagli,
anomalie -- con le coordinate tarate su UNO scatto: un pixel dentro `IMG_0008_2.tif` e la
sua posizione geografica. Composto il mosaico, quel pixel non esiste piu': serve la stessa
riga con i valori tarati sul mosaico, ed e' quello che si scrive qui, in
`output/<nome>_mosaic.csv`. Gli originali non vengono toccati.

**Si passa per la posa raffinata, non per la latitudine e longitudine dichiarate.** Sono
due strade diverse e portano a punti diversi. La seconda darebbe una coordinata
geograficamente corretta, ma sul mosaico cadrebbe ACCANTO alla feature, sfalsata di quanto
il raffinamento globale ha spostato quell'immagine rispetto a dove il GPS la metteva -- che
e' esattamente cio' che tutta la pipeline serve a stimare. La prima invece segue l'immagine
dov'e' finita davvero. Lo scarto fra le due e' una misura interessante di per se', e finisce
nel riepilogo.

La catena e' tutta composizione di trasformazioni gia' calcolate altrove:

    (X, Y) pixel del file consegnato
      -> est/nord proiettati        record["source_to_utm"]     preprocessing/
      -> pixel del fotogramma       inv(record["frame_to_utm"]) preprocessing/
      -> pixel del mosaico          posa traslata               utils.poses, utils.mosaic
      -> est/nord proiettati        origine del canvas e GSD    utils.georef
      -> lon/lat                    to_wgs84                    utils.geodesy

Le righe che portano solo lat/lon entrano dal secondo passo invece che dal primo.

Una sorgente che non sappia collocare da sola i pixel dei propri scatti a terra -- perche'
la consegna non e' georeferenziata -- non espone `frame_to_utm`, e qui non si rimappa nulla.
Non e' un caso da coprire con un ripiego: senza quella trasformazione un punto
sull'immagine non ha una posizione, e inventargliela sarebbe peggio che dire di no.
"""
import csv
import io
import re
from pathlib import Path

import numpy as np
import pyproj

# Nomi di colonna riconosciuti, minuscoli. Non e' uno standard, e' quello che si e' visto:
# vanno allargati quando arriva un CSV scritto da qualcun altro.
COLONNE_IMMAGINE = ("path", "img_path", "image", "image_path", "file", "filename", "source")
COLONNE_X = ("x", "col", "column", "px")
COLONNE_Y = ("y", "row", "riga", "py")


def _cerca(campi, nomi) -> str | None:
    for campo in campi:
        if campo and campo.strip().lower() in nomi:
            return campo
    return None


def _cerca_prefisso(campi, prefisso: str) -> str | None:
    for campo in campi:
        if campo and campo.strip().lower().startswith(prefisso):
            return campo
    return None


def _leggi(path: Path):
    """Righe del CSV, piu' delimitatore e fine riga cosi' da poterli riprodurre.

    Il dialetto va conservato e non normalizzato: `subsoil_data.csv` e' separato da virgola,
    `mine.csv` da punto e virgola e ha una prima colonna senza nome. Riscriverli entrambi
    "puliti" romperebbe chi li legge.
    """
    grezzo = path.read_bytes().decode("utf-8-sig")
    fine_riga = "\r\n" if "\r\n" in grezzo else "\n"
    prima = grezzo.splitlines()[0] if grezzo.strip() else ""
    delimitatore = max(",;\t|", key=prima.count) if prima else ","

    lettore = csv.DictReader(io.StringIO(grezzo, newline=""), delimiter=delimitatore)
    return list(lettore.fieldnames or []), list(lettore), delimitatore, fine_riga


def _indice_per_nome(sorgente, tenuti: list[int]) -> dict[str, int]:
    """Da un nome di file citato in un CSV alla posizione dello scatto fra i `tenuti`.

    Le chiavi comprendono sia il file consegnato sia il fotogramma prodotto, senza
    estensione: un CSV puo' citare l'uno o l'altro, e nessuno dei due e' piu' legittimo.
    """
    tavola: dict[str, int] = {}
    for posizione, indice in enumerate(tenuti):
        record = sorgente.records[indice]
        for chiave in (record.get("source"), record.get("path")):
            if chiave is not None:
                tavola[Path(chiave).stem.lower()] = posizione
    return tavola


def _posizione_di(riferimento: str, tavola: dict[str, int]) -> int | None:
    """Lo scatto a cui si riferisce un percorso citato in un CSV, se e' nel mosaico.

    I CSV possono citare un RITAGLIO invece dello scatto -- `app/crop/IMG_0011_2_0.jpg` sta
    per `IMG_0011_2` -- quindi si prova il nome intero e poi lo si accorcia di un suffisso
    numerico alla volta: il primo che corrisponde vince, e non si continua a tagliare, per
    non trasformare `IMG_0011_2` in `IMG_0011`.
    """
    nome = Path(str(riferimento).replace("\\", "/")).stem.lower()
    while nome:
        if nome in tavola:
            return tavola[nome]
        accorciato = re.sub(r"_\d+$", "", nome)
        if accorciato == nome:
            return None
        nome = accorciato
    return None


def _fotogramma_su_cui_misurare(in_utm, preferito, inverse, image_size):
    """Su quale scatto leggere il punto, e dove ci cade. Ritorna (posizione, px, spostato).

    Normalmente e' lo scatto che il CSV cita. Il pre-pass pero' ritaglia qualche punto
    percento ai bordi -- e' il prezzo di fotogrammi senza nodata -- quindi un punto misurato
    sull'orlo dell'immagine consegnata puo' cadere appena fuori dal fotogramma
    corrispondente. Il mosaico lo copre lo stesso, ma attraverso lo scatto accanto: meglio
    prendere la posa di quello che estrapolare quella citata oltre i suoi bordi.
    """
    w, h = image_size

    def dove(posizione):
        px = inverse[posizione] @ in_utm
        return min(px[0], w - px[0], px[1], h - px[1]), px

    margine, px = dove(preferito)
    if margine >= 0:
        return preferito, px, False

    migliore = max(inverse, key=lambda p: dove(p)[0])
    margine_migliore, px_migliore = dove(migliore)
    if margine_migliore < 0:
        return preferito, px, False  # nessuno lo contiene: si tiene il citato
    return migliore, px_migliore, True


def remap_folder(
    sorgente,
    tenuti: list[int],
    transforms: list[np.ndarray],
    image_size: tuple[int, int],
    canvas_origin_m: tuple[float, float],
    gsd: float,
    to_wgs84,
    output_dir: Path,
    mosaic_name: str,
) -> list[str]:
    """Riscrive ogni CSV della cartella del volo con i valori tarati sul mosaico.

    `transforms` sono le pose gia' traslate sul canvas finale, alla risoluzione di
    `image_size`, nell'ordine di `tenuti`: le stesse che compongono il mosaico, quindi per
    costruzione i punti finiscono dove finiscono i pixel.

    Ritorna una riga di riepilogo per file, da stampare.
    """
    csv_sorgenti = sorted(
        p for p in Path(sorgente.source_dir).glob("*.csv") if not p.stem.endswith("_mosaic")
    )
    if not csv_sorgenti:
        return []
    if "frame_to_utm" not in sorgente.records[0]:
        return [
            f"{len(csv_sorgenti)} file .csv non rimappati: la sorgente {sorgente.kind!r} non "
            "colloca i pixel dei singoli scatti a terra, quindi non c'e' da dove partire"
        ]

    tavola = _indice_per_nome(sorgente, tenuti)
    inverse = {
        posizione: np.linalg.inv(sorgente.records[indice]["frame_to_utm"])
        for posizione, indice in enumerate(tenuti)
    }
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", sorgente.utm_crs, always_xy=True)
    origine_est, origine_nord = canvas_origin_m

    riepiloghi = []
    for path in csv_sorgenti:
        campi, righe, delimitatore, fine_riga = _leggi(path)
        if not campi or not righe:
            riepiloghi.append(f"{path.name}: vuoto, saltato")
            continue

        colonna_immagine = _cerca(campi, COLONNE_IMMAGINE)
        colonna_x = _cerca(campi, COLONNE_X)
        colonna_y = _cerca(campi, COLONNE_Y)
        colonna_lon = _cerca_prefisso(campi, "lon")
        colonna_lat = _cerca_prefisso(campi, "lat")
        ha_pixel = colonna_x is not None and colonna_y is not None
        ha_gradi = colonna_lon is not None and colonna_lat is not None
        if colonna_immagine is None or not (ha_pixel or ha_gradi):
            riepiloghi.append(
                f"{path.name}: saltato, servono i pixel (X, Y) oppure lon/lat, e una colonna "
                f"che dica da quale immagine vengono (trovate: "
                f"{', '.join(c for c in campi if c)})"
            )
            continue

        uscita, scartate, spostamenti, spostati = [], [], [], 0
        for riga in righe:
            posizione = _posizione_di(riga.get(colonna_immagine, ""), tavola)
            if posizione is None:
                scartate.append(riga.get(colonna_immagine, ""))
                continue

            record = sorgente.records[tenuti[posizione]]
            if ha_pixel:
                in_utm = record["source_to_utm"] @ np.array(
                    [float(riga[colonna_x]), float(riga[colonna_y]), 1.0]
                )
            else:
                est, nord = to_utm.transform(
                    float(riga[colonna_lon]), float(riga[colonna_lat])
                )
                in_utm = np.array([est, nord, 1.0])

            posizione, nel_fotogramma, spostato = _fotogramma_su_cui_misurare(
                in_utm, posizione, inverse, image_size
            )
            spostati += int(spostato)

            nel_mosaico = transforms[posizione] @ nel_fotogramma
            est_m = origine_est + nel_mosaico[0] * gsd
            nord_m = origine_nord - nel_mosaico[1] * gsd
            lon, lat = to_wgs84.transform(est_m, nord_m)
            spostamenti.append(float(np.hypot(est_m - in_utm[0], nord_m - in_utm[1])))

            nuova = dict(riga)
            if ha_pixel:
                nuova[colonna_x] = f"{nel_mosaico[0]:.1f}"
                nuova[colonna_y] = f"{nel_mosaico[1]:.1f}"
                # Con i pixel riferiti al mosaico, lasciare il nome dello scatto sarebbe una
                # trappola: X e Y non stanno piu' dentro quell'immagine.
                nuova[colonna_immagine] = mosaic_name
            if ha_gradi:
                nuova[colonna_lon] = f"{lon:.10f}"
                nuova[colonna_lat] = f"{lat:.10f}"
            uscita.append(nuova)

        destinazione = Path(output_dir) / f"{path.stem}_mosaic.csv"
        with open(destinazione, "w", newline="", encoding="utf-8") as f:
            scrittore = csv.DictWriter(
                f, fieldnames=campi, delimiter=delimitatore, lineterminator=fine_riga
            )
            scrittore.writeheader()
            scrittore.writerows(uscita)

        note = []
        if spostamenti:
            note.append(
                f"il raffinamento li ha spostati di {np.median(spostamenti):.3f} m di "
                f"mediana, {max(spostamenti):.3f} m di massimo"
            )
        if spostati:
            note.append(f"{spostati} letti sullo scatto accanto, fuori dal proprio")
        if scartate:
            note.append(
                f"{len(scartate)} scartati, immagine non nel mosaico "
                f"(il primo e' {scartate[0]!r})"
            )
        riepiloghi.append(
            f"{destinazione.name}: {len(uscita)}/{len(righe)} punti"
            + (" | " + " | ".join(note) if note else "")
        )
    return riepiloghi
