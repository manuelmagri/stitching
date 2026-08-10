"""Preprocessing delle immagini drone DJI.

Prepara i tre file che la pipeline consuma, piu' le immagini rettificate:

    genera_calibrazione(immagine_riferimento, output_file)
        -> data/calibration.json    intrinseci e dimensione delle immagini rettificate

    genera_metadati(image_folder, output_file, exiftool_path, ...)
        -> data/metadata.json       GPS, quota e assetto di ogni scatto

    genera_traslazioni(metadata_file, output_file)
        -> data/translations.json   delta [est, nord] inter-frame, in metri

    correggi_distorsione_cartella(input_dir, output_dir, exiftool_path)
        -> cartella di .jpg rettificate, e' l'input di main.py

Vincoli d'ordine: le traslazioni richiedono i metadati gia' scritti; la calibrazione
va rigenerata ogni volta che si rifa' la rettifica, perche' descrive le immagini
prodotte da quella (focale, centro ottico e dimensione cambiano col ritaglio).
Metadati e rettifica sono invece indipendenti fra loro.

Ogni modulo e' eseguibile da solo. Va invocato con -m e dalla radice del progetto,
perche' gli import interni sono assoluti (`preprocessing.<modulo>`):

    python -m preprocessing.create_calibration <scatto_originale.JPG>
    python -m preprocessing.create_metadata <cartella_volo>
    python -m preprocessing.create_translations
    python -m preprocessing.undistort_image

`_xmp` e' un helper interno (intrinseci XMP e parametri di rettifica), non fa parte
dell'API.
"""
import importlib as _importlib

# I quattro nomi sopra sono raggiungibili anche da qui (`from preprocessing import
# genera_calibrazione`), ma vengono risolti solo quando si usano, non all'import del
# package. Importarli subito renderebbe ogni `python -m preprocessing.<modulo>` un
# doppio caricamento -- una volta come sottomodulo tirato dentro da qui, una come
# __main__ -- e runpy lo segnala con un RuntimeWarning a ogni invocazione.
_API = {
    "genera_calibrazione": "preprocessing.create_calibration",
    "genera_metadati": "preprocessing.create_metadata",
    "genera_traslazioni": "preprocessing.create_translations",
    "correggi_distorsione_cartella": "preprocessing.undistort_image",
}


def __getattr__(name):
    modulo = _API.get(name)
    if modulo is None:
        raise AttributeError(f"il package {__name__!r} non espone {name!r}")
    return getattr(_importlib.import_module(modulo), name)


def __dir__():
    return sorted([*globals(), *_API])
