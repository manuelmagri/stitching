import os
import xml.etree.ElementTree as ET
import numpy as np

DJI_NS = "{http://www.dji.com/drone-dji/1.0/}"

def analizza_xmp(file_path):
    """Estrae il blocco XMP grezzo da un file immagine DJI."""
    with open(file_path, 'rb') as file:
        data = file.read()
    start_xmp = data.find(b"<x:xmpmeta")
    end_xmp = data.find(b"</x:xmpmeta>")
    if start_xmp == -1 or end_xmp == -1:
        raise ValueError(f"Dati XMP non trovati in {file_path}")
    end_xmp += len(b"</x:xmpmeta>")
    return data[start_xmp:end_xmp].decode('utf-8')


def estrai_matrice_intrinseca(xmp_string):
    """Estrae intrinseci di camera e coefficienti di distorsione dall'XMP.

    Ritorna (focal_length, cx, cy, dist_coeffs) dove dist_coeffs e'
    un np.array([k1, k2, p1, p2, k3]) pronto per cv2.undistort.
    """
    root = ET.fromstring(xmp_string)
    focal_length = cx = cy = dewarp = None

    for child in root.iter():
        for key, val in child.attrib.items():
            if key == DJI_NS + "CalibratedFocalLength":
                focal_length = float(val)
            elif key == DJI_NS + "CalibratedOpticalCenterX":
                cx = float(val)
            elif key == DJI_NS + "CalibratedOpticalCenterY":
                cy = float(val)
            elif key == DJI_NS + "DewarpData":
                dewarp = val

    if None in (focal_length, cx, cy) or dewarp is None:
        raise ValueError("Campi DJI per intrinseci/distorsione non trovati nell'XMP")

    # DewarpData: "data;fx,fy,cx_off,cy_off,k1,k2,p1,p2,k3"
    parts = dewarp.split(";", 1)[1].split(",")
    dist_coeffs = np.array([float(v) for v in parts[4:9]])

    return focal_length, cx, cy, dist_coeffs


def leggi_intrinseci_da_immagine(file_path):
    """Scorciatoia: apre l'immagine, legge l'XMP e ne estrae gli intrinseci."""
    return estrai_matrice_intrinseca(analizza_xmp(file_path))


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    drone = "DJI_202604161249_001_UgCS-Create-Area-Route3"
    file = "DJI_20260416125617_0001_D.JPG"
    file_path = os.path.join(project_root, "immagini", "immagini_drone", drone, file)

    focal, cx, cy, dist = leggi_intrinseci_da_immagine(file_path)
    print(f"Focal length: {focal}")
    print(f"Centro ottico: ({cx}, {cy})")
    print(f"Coefficienti distorsione: {dist}")
