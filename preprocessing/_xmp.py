"""Helper interno: intrinseci di camera letti dal blocco XMP DJI, e parametri di rettifica.

Non fa parte dell'API pubblica del package. Lo usano `create_calibration` (per scrivere
data/calibration.json) e `undistort_image` (per correggere la distorsione), che hanno
entrambi bisogno degli stessi parametri, senza passare da una calibrazione a scacchiera.

Il calcolo della rettifica sta qui, e non duplicato nei due chiamanti, perche' se i due
lo facessero per conto proprio la calibrazione scritta su disco potrebbe descrivere
immagini diverse da quelle effettivamente prodotte.
"""
import xml.etree.ElementTree as ET

import cv2
import numpy as np

DJI_NS = "{http://www.dji.com/drone-dji/1.0/}"

def _blocco_xmp(image_path):
    """Blocco XMP grezzo estratto dai byte di un file immagine DJI."""
    with open(image_path, "rb") as f:
        data = f.read()
    start = data.find(b"<x:xmpmeta")
    end = data.find(b"</x:xmpmeta>")
    if start == -1 or end == -1:
        raise ValueError(f"Dati XMP non trovati in {image_path}")
    return data[start:end + len(b"</x:xmpmeta>")].decode("utf-8")


def leggi_intrinseci(image_path):
    """(camera_matrix 3x3, dist_coeffs) letti dall'XMP di `image_path`.

    I coefficienti escono come np.array([k1, k2, p1, p2, k3]), l'ordine atteso
    da cv2.undistort. Sono identici per tutte le foto della stessa camera,
    quindi basta leggerli da un'immagine di riferimento.
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
        raise ValueError(f"Campi DJI per intrinseci/distorsione assenti nell'XMP di {image_path}")

    camera_matrix = np.array([
        [focal, 0.0, cx],
        [0.0, focal, cy],
        [0.0, 0.0, 1.0],
    ])
    # DewarpData: "data;fx,fy,cx_off,cy_off,k1,k2,p1,p2,k3"
    dist_coeffs = np.array([float(v) for v in dewarp.split(";", 1)[1].split(",")[4:9]])
    return camera_matrix, dist_coeffs


# Quanto del fotogramma originale sopravvive alla rettifica. Con alpha=1 cv2 conserva
# tutti i pixel sorgente -- l'immagine rettificata ha bordi curvi vuoti -- e `roi` e' il
# piu' grande rettangolo interamente valido, al quale si ritaglia. Cambiare questo valore
# cambia focale, centro ottico e dimensione delle immagini prodotte.
ALPHA_RETTIFICA = 1


def parametri_rettifica(image_path):
    """Parametri per rettificare le foto di questa camera, dedotti da `image_path`.

    `image_path` deve essere uno scatto ORIGINALE (ancora distorto): da li' si leggono
    gli intrinseci nell'XMP e la dimensione dal file. Sono identici per tutte le foto
    della stessa camera, quindi una sola immagine basta per l'intero volo.

    Ritorna un dict con:
      - camera_matrix, dist_coeffs : intrinseci originali, argomenti di cv2.undistort
      - new_camera_matrix, roi     : nuova matrice e rettangolo di ritaglio (x, y, w, h)
      - camera_matrix_rettificata  : intrinseci EFFETTIVI dell'immagine rettificata e
                                     ritagliata, cioe' quelli che descrivono i file prodotti
      - image_size                 : (larghezza, altezza) dell'immagine prodotta
      - image_size_originale       : (larghezza, altezza) dello scatto di partenza
    """
    camera_matrix, dist_coeffs = leggi_intrinseci(image_path)

    sample = cv2.imread(str(image_path))
    if sample is None:
        raise RuntimeError(f"Impossibile leggere {image_path}")
    h, w = sample.shape[:2]

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
