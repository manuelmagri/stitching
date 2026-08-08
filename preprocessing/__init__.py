"""Preprocessing delle immagini drone DJI.

Prepara i tre file che la pipeline consuma, piu' le immagini non distorte:

    create_calib.genera_calibrazione(immagine_riferimento, output_file)
        -> data/calibration.json    camera matrix letta dall'XMP DJI

    create_metadata.genera_metadati(image_folder, output_file, exiftool_path, ...)
        -> data/metadati.json       GPS, quota e assetto di ogni scatto

    create_translations.genera_traslazioni(metadata_file, output_file)
        -> data/translations.json   delta [est, nord] inter-frame, in metri

    undistort_image.correggi_distorsione_cartella(input_dir, output_dir, exiftool_path)
        -> cartella di .jpg non distorte, e' l'input di main.py

Calibrazione e metadati sono indipendenti; le traslazioni richiedono i metadati
gia' scritti. Ogni modulo e' eseguibile da solo sui percorsi di default:

    python -m preprocessing.create_calib
    python -m preprocessing.create_metadata
    python -m preprocessing.create_translations
    python -m preprocessing.undistort_image

`_xmp` e' un helper interno (intrinseci letti dall'XMP), non fa parte dell'API.
"""
