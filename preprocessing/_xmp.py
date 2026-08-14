"""Helper interno: intrinseci di camera dal blocco XMP DJI, e parametri di rettifica.

Espone una funzione sola, `parametri_rettifica`. La usano `create_calibration` (per
scrivere data/calibration.json) e `undistort_images` (per correggere la distorsione):
il calcolo sta qui e non duplicato nei due, perche' se ciascuno lo facesse per conto
proprio la calibrazione su disco potrebbe descrivere immagini diverse da quelle
effettivamente prodotte.
"""
import xml.etree.ElementTree as ET

import cv2
import numpy as np

DJI_NS = "{http://www.dji.com/drone-dji/1.0/}"

# Quanto del fotogramma originale sopravvive alla rettifica. Con alpha=1 cv2 conserva
# tutti i pixel sorgente -- l'immagine rettificata ha bordi curvi vuoti -- e `roi` e' il
# piu' grande rettangolo interamente valido, al quale si ritaglia. Cambiare questo valore
# cambia focale, centro ottico e dimensione delle immagini prodotte.
ALPHA_RETTIFICA = 1


def _blocco_xmp(image_path) -> str:
    """Blocco XMP grezzo estratto dai byte di un file immagine DJI."""
    with open(image_path, "rb") as f:
        data = f.read()
    inizio = data.find(b"<x:xmpmeta")
    fine = data.find(b"</x:xmpmeta>")
    if inizio == -1 or fine == -1:
        raise ValueError(
            f"Dati XMP non trovati in {image_path}. Se e' un'immagine gia' rettificata, "
            "il blocco XMP non c'e' piu': questo passo vuole gli scatti ORIGINALI."
        )
    return data[inizio : fine + len(b"</x:xmpmeta>")].decode("utf-8")


def _leggi_intrinseci(image_path):
    """(camera_matrix 3x3, dist_coeffs) letti dall'XMP di `image_path`.

    I coefficienti escono come np.array([k1, k2, p1, p2, k3]), l'ordine atteso da
    cv2.undistort.
    """
    focal = cx = cy = dewarp = None
    for child in ET.fromstring(_blocco_xmp(image_path)).iter():
        for key, val in child.attrib.items():
            if key == DJI_NS + "CalibratedFocalLength":
                focal = float(val)
            elif key == DJI_NS + "CalibratedOpticalCenterX":
                cx = float(val)
            elif key == DJI_NS + "CalibratedOpticalCenterY":
                cy = float(val)
            elif key == DJI_NS + "DewarpData":
                dewarp = val

    if None in (focal, cx, cy, dewarp):
        raise ValueError(
            f"Campi DJI per intrinseci/distorsione assenti nell'XMP di {image_path}"
        )

    camera_matrix = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]])
    # DewarpData: "data;fx,fy,cx_off,cy_off,k1,k2,p1,p2,k3"
    dist_coeffs = np.array([float(v) for v in dewarp.split(";", 1)[1].split(",")[4:9]])
    return camera_matrix, dist_coeffs


def parametri_rettifica(image_path) -> dict:
    """Parametri per rettificare le foto di questa camera, dedotti da `image_path`.

    `image_path` deve essere uno scatto ORIGINALE (ancora distorto): da li' si leggono
    gli intrinseci nell'XMP e la dimensione dal file. Sono identici per tutte le foto
    della stessa camera, quindi una sola immagine basta per l'intero volo.

        camera_matrix, dist_coeffs  intrinseci originali, argomenti di cv2.undistort
        new_camera_matrix, roi      nuova matrice e rettangolo di ritaglio (x, y, w, h)
        camera_matrix_rettificata   intrinseci EFFETTIVI dell'immagine prodotta
        image_size                  (larghezza, altezza) dell'immagine prodotta
        image_size_originale        (larghezza, altezza) dello scatto di partenza
    """
    camera_matrix, dist_coeffs = _leggi_intrinseci(image_path)

    campione = cv2.imread(str(image_path))
    if campione is None:
        raise RuntimeError(f"Impossibile leggere {image_path}")
    h, w = campione.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), ALPHA_RETTIFICA, (w, h)
    )
    x, y, rw, rh = (int(v) for v in roi)

    # Il ritaglio sposta l'origine dei pixel: il centro ottico va traslato di (x, y),
    # la focale invece resta quella di new_camera_matrix.
    camera_matrix_rettificata = new_camera_matrix.copy()
    camera_matrix_rettificata[0, 2] -= x
    camera_matrix_rettificata[1, 2] -= y

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "new_camera_matrix": new_camera_matrix,
        "roi": (x, y, rw, rh),
        "camera_matrix_rettificata": camera_matrix_rettificata,
        "image_size": (rw, rh),
        "image_size_originale": (w, h),
    }
