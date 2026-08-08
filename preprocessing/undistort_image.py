import glob
import os
import subprocess
import cv2

from preprocessing._xmp import leggi_intrinseci

def correggi_distorsione_cartella(input_dir, output_dir, exiftool_path):
    """Corregge la distorsione di tutte le .jpg di `input_dir` salvandole in `output_dir`.

    Mantiene il nome originale del file (necessario per ricollegarlo ai
    metadati IMU) e copia le coordinate GPS dall'originale.
    """
    immagini = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    if not immagini:
        raise FileNotFoundError(f"Nessuna .jpg in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # Gli intrinseci sono identici per tutte le foto della stessa camera:
    # leggiamo l'XMP della prima e calcoliamo (newK, roi) una volta sola.
    camera_matrix, dist_coeffs = leggi_intrinseci(immagini[0])

    sample = cv2.imread(immagini[0])
    if sample is None:
        raise RuntimeError(f"Impossibile leggere {immagini[0]}")
    h, w = sample.shape[:2]

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )
    x, y, rw, rh = roi

    output_paths = []
    for src_path in immagini:
        img = cv2.imread(src_path)
        if img is None:
            print(f"Saltata (lettura fallita): {src_path}")
            continue

        undistorted = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)
        undistorted = undistorted[y:y + rh, x:x + rw]

        dst_path = os.path.join(output_dir, os.path.basename(src_path))
        cv2.imwrite(dst_path, undistorted)
        output_paths.append(dst_path)
        print(f"Salvata: {dst_path}")

    # Propaga le tag GPS dall'originale: %f.%e lato sorgente lega ogni
    # output al file omonimo nella cartella di input. Va spezzato in chunk
    # per non superare il limite di lunghezza della command line di Windows.
    chunk_size = 30
    for i in range(0, len(output_paths), chunk_size):
        chunk = output_paths[i:i + chunk_size]
        subprocess.run(
            [exiftool_path,
             "-TagsFromFile", os.path.join(input_dir, "%f.%e"),
             "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
             "-overwrite_original",
             *chunk],
            check=True,
        )


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    drone = "DJI_202604161249_001_UgCS-Create-Area-Route3"
    input_dir = os.path.join(project_root, "immagini", "immagini_drone", drone)
    output_dir = os.path.join(project_root, "immagini", "immagini_drone", "immagini_senza_distorsione")
    exiftool_path = os.path.join(project_root, "exiftool-13.53_64", "exiftool.exe")

    correggi_distorsione_cartella(input_dir, output_dir, exiftool_path)