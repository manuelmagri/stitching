import json
import os
import numpy as np

from extract_xmp import leggi_intrinseci_da_immagine


def matrice_rotazione(roll_deg, pitch_deg, yaw_deg):
    """Matrice di rotazione 3x3 da angoli gimbal in gradi (R_z @ R_y @ R_x)."""
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return R_z @ R_y @ R_x


def salva_calibrazione(output_dir, camera_matrix, dist_coeffs, extr_matrix, proj_matrix):
    "Salva il file di calibrazione nella cartella 'data/'."
    os.makedirs(output_dir, exist_ok=True)
    files = {
        "cameraMatrix.txt": camera_matrix.tolist(),
        "dist.txt": dist_coeffs.tolist(),
        "extrMatrix.txt": extr_matrix.tolist(),
        "projMatrix.txt": proj_matrix.tolist(),
    }
    for nome, dati in files.items():
        with open(os.path.join(output_dir, nome), "w") as f:
            json.dump(dati, f, indent=2)
    with open(os.path.join(output_dir, "calibration.txt"), "w") as f:
        json.dump({"cameraMatrix": camera_matrix.tolist(),
                   "projMatrix": proj_matrix.tolist()}, f, indent=2)


def genera_calibrazione(immagine_riferimento, output_dir,
                        gimbal_roll_deg=180.00,
                        gimbal_pitch_deg=-89.90,
                        gimbal_yaw_deg=89.80):
    """Costruisce e salva la calibrazione di camera per `immagine_riferimento`.

    Gli intrinseci sono letti dall'XMP dell'immagine; l'estrinseca e' la
    rotazione gimbal (traslazione nulla). Ritorna la 4-tupla
    (camera_matrix, dist_coeffs, extr_matrix, proj_matrix).
    """
    focal, cx, cy, dist_coeffs = leggi_intrinseci_da_immagine(immagine_riferimento)

    camera_matrix = np.array([
        [focal, 0, cx],
        [0, focal, cy],
        [0, 0, 1],
    ])

    R = matrice_rotazione(gimbal_roll_deg, gimbal_pitch_deg, gimbal_yaw_deg)
    t = np.zeros((3, 1))
    extr_matrix = np.hstack((R, t))
    proj_matrix = camera_matrix @ extr_matrix

    salva_calibrazione(output_dir, camera_matrix, dist_coeffs, extr_matrix, proj_matrix)
    return camera_matrix, dist_coeffs, extr_matrix, proj_matrix


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    drone = "DJI_202604161249_001_UgCS-Create-Area-Route3"
    file = "DJI_20260416125617_0001_D.JPG"
    immagine = os.path.join(project_root, "immagini", "immagini_drone", drone, file)
    output_dir = os.path.join(project_root, "data")

    genera_calibrazione(immagine, output_dir)
