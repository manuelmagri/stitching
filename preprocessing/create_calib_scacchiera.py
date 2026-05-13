import glob
import json
import os
import cv2 as cv
import numpy as np


def trova_angoli_scacchiera(immagini_dir, chessboard_size, square_size_mm):
    """Trova gli angoli della scacchiera in tutte le .jpg di `immagini_dir`.

    Ritorna (objpoints, imgpoints) compatibili con cv2.calibrateCamera.
    """
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp = objp * square_size_mm

    objpoints = []
    imgpoints = []

    for percorso in glob.glob(os.path.join(immagini_dir, "*.jpg")):
        img = cv.imread(percorso)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        if not ret:
            continue
        objpoints.append(objp)
        corners_raffinati = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners_raffinati)

    return objpoints, imgpoints


def calcola_proiezioni(camera_matrix, rvecs, tvecs):
    """Da rvec/tvec di calibrateCamera ricostruisce [R|t] e P = K @ [R|t] per ogni vista."""
    extrinsics = []
    projections = []
    for rvec, tvec in zip(rvecs, tvecs):
        R, _ = cv.Rodrigues(rvec)
        extr = np.hstack((R, tvec))
        extrinsics.append(extr)
        projections.append(camera_matrix @ extr)
    return extrinsics, projections


def salva_calibrazione_scacchiera(output_dir, camera_matrix, dist_coeffs, extrinsics, projections):
    """Salva i risultati della calibrazione in `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cameraMatrix.txt"), "w") as f:
        json.dump(camera_matrix.tolist(), f, indent=2)
    with open(os.path.join(output_dir, "dist.txt"), "w") as f:
        json.dump(dist_coeffs.tolist(), f, indent=2)
    with open(os.path.join(output_dir, "extrMatrix.txt"), "w") as f:
        json.dump([E.tolist() for E in extrinsics], f, indent=2)
    with open(os.path.join(output_dir, "projMatrix.txt"), "w") as f:
        json.dump([P.tolist() for P in projections], f, indent=2)
    with open(os.path.join(output_dir, "calibration.txt"), "w") as f:
        json.dump({"cameraMatrix": camera_matrix.tolist(),
                   "projMatrix": projections[0].tolist()}, f, indent=2)


def calibra_da_scacchiera(immagini_dir, output_dir,
                          chessboard_size=(6, 9),
                          frame_size=(5280, 3956),
                          square_size_mm=25):
    """Calibra la camera dalle foto di una scacchiera e salva i risultati.

    Ritorna (camera_matrix, dist_coeffs, extrinsics, projections).
    """
    objpoints, imgpoints = trova_angoli_scacchiera(immagini_dir, chessboard_size, square_size_mm)
    if not objpoints:
        raise RuntimeError(f"Nessun angolo trovato nelle .jpg di {immagini_dir}")

    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, frame_size, None, None
    )

    extrinsics, projections = calcola_proiezioni(camera_matrix, rvecs, tvecs)

    for i, P in enumerate(projections):
        print(f"Matrice di Proiezione {i}:")
        print(P)

    salva_calibrazione_scacchiera(output_dir, camera_matrix, dist_coeffs, extrinsics, projections)
    return camera_matrix, dist_coeffs, extrinsics, projections


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    immagini_dir = os.path.join(project_root, "immagini", "immagini_scacchiera_drone")
    output_dir = os.path.join(project_root, "data")

    calibra_da_scacchiera(immagini_dir, output_dir)
