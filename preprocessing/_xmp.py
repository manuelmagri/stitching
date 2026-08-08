"""Helper interno: intrinseci di camera letti dal blocco XMP DJI.

Non fa parte dell'API pubblica del package. Lo usano `create_calib` (per scrivere
data/calibration.json) e `undistort_image` (per correggere la distorsione), che
hanno entrambi bisogno della stessa coppia (camera matrix, coefficienti di
distorsione) senza passare da una calibrazione a scacchiera.
"""
import xml.etree.ElementTree as ET

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
