"""Il contratto fra "com'e' stato consegnato il volo" e "come si cuce".

La pipeline ha una frattura netta a meta': da `utils.pairing` in poi nulla sa come sono
nate le pose -- si vedono solo impronte, pixel e trasformazioni. Tutto cio' che sta prima
serve a capire il volo, e dipende invece dal formato in cui e' arrivato.

`Source` e' quella frattura resa esplicita. Un lettore la riempie, `main.py` la consuma, e
in mezzo non c'e' nessun ramo: due formati di consegna, due lettori, una pipeline sola.

    utils.source_exif    immagini rettificate + telemetria EXIF/XMP (i voli DJI)
    utils.source_ortho   una ortofoto per scatto, gia' georeferenziata (tag GDAL SHOT_*)

Il contratto sui fotogrammi -- rettangolari, nadirali, a pixel quadrati, tutti della stessa
dimensione -- non e' scritto qui perche' non e' verificabile qui: e' quello che
`preprocessing.undistort_images` e `preprocessing.create_ortho_frames` producono, ciascuno
partendo dalla propria consegna.

REGOLA. Nessun parametro della pipeline puo' dipendere da `kind`. Se un valore differisce
fra due sorgenti deve essere CALCOLATO da qualcosa di misurabile -- la dimensione del
fotogramma, l'impronta a terra, la distribuzione delle sovrapposizioni -- mai deciso in
base a chi ha prodotto i file. I due booleani qui sotto sono l'unica eccezione, e non sono
parametri: sono affermazioni su cosa sono i dati, e cambiano solo quello che si stampa.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Source:
    """Un volo pronto per lo stitching, qualunque sia il formato da cui viene."""

    kind: str
    source_dir: Path
    """La cartella che l'utente ha indicato. Serve a ritrovare cio' che sta accanto alle
    immagini e non e' un'immagine -- per esempio i CSV di punti che `utils.points` rimappa."""

    records: list[dict]
    """Uno per scatto, in ordine di frame. Campi che la pipeline legge:

        index              numero di frame, per --da/--a e per le etichette
        path               il file da cui leggere i pixel
        lat, lon           DOVE IL PRODOTTO DICE CHE SIA lo scatto. Non partecipa allo
                           stitching: e' il bersaglio del fit finale in `utils.georeference`
        flight_yaw_deg     rotta in gradi da nord, oraria
        gimbal_pitch_deg   opzionale. Se manca, `flight.mark_curves` non applica il criterio
                           del nadir -- ed e' corretto che sia cosi' su immagini gia'
                           ortorettificate, dove la premessa del criterio non vale

    Due campi facoltativi servono solo a `utils.points`, e ci sono quando la consegna
    permette di collocare un pixel a terra senza passare dallo stitching:

        frame_to_utm       affine 3x3 dai pixel del fotogramma alle coordinate proiettate
        source_to_utm      idem dai pixel del file COME CONSEGNATO, che puo' non essere il
                           fotogramma su cui la pipeline lavora
    """

    positions_m: np.ndarray
    """(N, 2) est/nord in metri, origine sul primo scatto. E' il SEED della traslazione."""

    gsds: np.ndarray
    """(N,) metri per pixel di ogni scatto, a piena risoluzione."""

    image_size: tuple[int, int]

    utm_crs: str
    """Il CRS proiettato in cui `positions_m` e gli eventuali `*_to_utm` sono espressi."""

    grid_north: bool
    """Se `positions_m` e `flight_yaw_deg` siano riferiti al nord GRIGLIA anziche' al nord
    VERO. La bussola misura il nord vero e UTM usa quello griglia: chi costruisce il frame
    locale sull'assetto lascia da assorbire al fit finale la convergenza del meridiano, chi
    lo costruisce direttamente in UTM no. Governa il valore atteso, non il fit."""

    gps_seeded: bool
    """Se il GPS abbia contribuito a `positions_m`. Quando e' falso il residuo del fit
    finale e' una validazione INDIPENDENTE della ricostruzione, perche' misura contro dati
    che non l'hanno prodotta; quando e' vero non lo e', e dirlo lo stesso sarebbe una
    frase falsa. Non cambia un solo calcolo: cambia cosa si puo' affermare del risultato."""

    intro: list[str] = field(default_factory=list)
    """Righe da stampare in fase di caricamento, specifiche della consegna."""


def detect(images_dir: Path) -> str:
    """Da che formato e' fatta la cartella, dall'estensione dei file che contiene."""
    suffissi = {p.suffix.lower() for p in Path(images_dir).iterdir() if p.is_file()}
    if suffissi & {".jpg", ".jpeg"}:
        return "exif"
    if suffissi & {".tif", ".tiff"}:
        return "ortho"
    raise ValueError(
        f"{images_dir} non contiene ne' .jpg ne' .tif: non riconosco il formato di "
        "consegna. Forzalo con --sorgente."
    )


def _modulo(kind: str):
    if kind == "exif":
        from utils import source_exif

        return source_exif
    if kind == "ortho":
        from utils import source_ortho

        return source_ortho
    raise ValueError(f"sorgente sconosciuta: {kind!r}. Usa 'exif', 'ortho' o 'auto'.")


def missing_inputs(images_dir: Path, kind: str = "auto") -> list[str]:
    """Cosa manca per poter caricare questo volo, in righe da stampare. Vuota se tutto c'e'.

    Va chiamata PRIMA di `load`: i file di appoggio si generano con `preprocessing/`, e
    accorgersene subito costa un istante mentre accorgersene a meta' caricamento costa
    l'attesa e un messaggio peggiore.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return [f"cartella mancante: {images_dir}"]
    try:
        kind = detect(images_dir) if kind == "auto" else kind
    except ValueError as errore:
        return [str(errore)]
    return _modulo(kind).missing_inputs(images_dir)


def load(
    images_dir: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    kind: str = "auto",
) -> Source:
    """Il volo in `images_dir`, ristretto ai frame richiesti."""
    images_dir = Path(images_dir)
    kind = detect(images_dir) if kind == "auto" else kind
    return _modulo(kind).load(images_dir, frame_start, frame_end)
