"""Preprocessing delle immagini drone DJI: lettura XMP, calibrazione,
correzione della distorsione, estrazione dei metadati EXIF/IMU.

API pubblica del package (importabile come `from preprocessing import X`):

    leggi_intrinseci_da_immagine(file_path)
        -> (focal, cx, cy, dist_coeffs) letti dall'XMP DJI

    analizza_xmp(file_path)
        -> stringa XMP grezza

    estrai_matrice_intrinseca(xmp_string)
        -> (focal, cx, cy, dist_coeffs) parsando un XMP gia' letto

    correggi_distorsione_cartella(input_dir, output_dir, exiftool_path)
        applica cv2.undistort a tutte le .jpg di una cartella

    genera_calibrazione(immagine_riferimento, output_dir, ...)
        calibrazione di camera con intrinseci letti dall'XMP

    calibra_da_scacchiera(immagini_dir, output_dir, ...)
        calibrazione di camera tramite scacchiera (path alternativo)

    estrai_metadati_da_immagini(image_folder, output_file, exiftool_path, ...)
        estrazione GPS / attitude / velocita' di volo via exiftool
"""

from preprocessing.extract_xmp import (
    analizza_xmp,
    estrai_matrice_intrinseca,
    leggi_intrinseci_da_immagine,
)
from preprocessing.undistort_image import correggi_distorsione_cartella
from preprocessing.create_calib import genera_calibrazione
from preprocessing.create_calib_scacchiera import calibra_da_scacchiera
from preprocessing.extract_metadata import estrai_metadati_da_immagini

__all__ = [
    "analizza_xmp",
    "estrai_matrice_intrinseca",
    "leggi_intrinseci_da_immagine",
    "correggi_distorsione_cartella",
    "genera_calibrazione",
    "calibra_da_scacchiera",
    "estrai_metadati_da_immagini",
]
