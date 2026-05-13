import pickle
import numpy as np
import math

width= 8192
height= 5460

# DJI Zenmuse P1
#calcolo larghezza e altezza del sensore
diag_sens_mm = 43.3  #43.18  # Diagonale del sensore in mm  rapporto d'aspettto 3:2
#x = diag_sens_mm / math.sqrt(13)  # Calcolo di x
#width_sens = 3 * x  # Larghezza
#height_sens = 2 * x  # Altezza
width_sens = 36.045  # Larghezza
height_sens = 24.024  # Altezza

# Lunghezza focale in mm (come da metadati)
focal_length_mm = 35

# Lunghezza focale in pixel
focal_length_x = (focal_length_mm * width) / width_sens
focal_length_y = (focal_length_mm * height) / height_sens
center_x = width / 2
center_y = height / 2
dist = np.array([0, 0, 0, 0, 0])  # nessuna distorsione, poiché i dati sono calibrati

# Matrice intrinseca
cameraMatrix = np.array([
    [focal_length_x, 0, center_x],
    [0, focal_length_y, center_y],
    [0, 0, 1]
])

# Angoli di rotazione in gradi (da convertire in radianti)
gimbal_roll_deg = +180.00       # Gimbal Roll Degree
gimbal_yaw_deg = +89.80      # Gimbal Yaw Degree
gimbal_pitch_deg = -89.90    # Gimbal Pitch Degree

# Conversione in radianti
roll = np.deg2rad(gimbal_roll_deg)
yaw = np.deg2rad(gimbal_yaw_deg)
pitch = np.deg2rad(gimbal_pitch_deg)

# Matrici di rotazione per roll, pitch e yaw
R_x = np.array([[1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]])

R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]])

R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]])

# Matrice di rotazione totale
R = R_z @ R_y @ R_x

# Vettore di traslazione (considerato nullo in questo esempio)
t = np.array([[0], [0], [0]])

# Matrice estrinseca [R | t]
extrMatrix = np.hstack((R, t))

# Matrice di proiezione P = K * [R | t]
projMatrix = cameraMatrix @ extrMatrix

pickle.dump(cameraMatrix, open( "data/cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "data/dist.pkl", "wb" ))
pickle.dump(extrMatrix, open( "data/extrMatrix.pkl", "wb" ))
pickle.dump(projMatrix, open( "data/projMatrix.pkl", "wb" ))
pickle.dump((cameraMatrix, projMatrix), open( "data/calibration.pkl", "wb" ))
