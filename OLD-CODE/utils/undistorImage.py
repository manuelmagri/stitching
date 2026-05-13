import cv2
import numpy as np
import glob


# Carica le immagini dalla cartella
immagini = glob.glob('C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/immagini/immagini_drone/DJI_202406171516_002_Arezzo/*.jpg')

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
dist_coeffs = np.array([0, 0, 0, 0, 0])  # nessuna distorsione, poiché i dati sono calibrati



# Matrice intrinseca
camera_matrix = np.array([
    [focal_length_x, 0, center_x],
    [0, focal_length_y, center_y],
    [0, 0, 1]
])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))

# Parametri di ritaglio per eliminare bordi neri
crop_margin = 0  # Margine da ritagliare agli angoli (adatta il valore)

# Elaborazione delle immagini
i = 0
for immagine in sorted(immagini):
    if immagine is not None:
        img = cv2.imread(immagine)
        #img = cv2.bilateralFilter(img, 3, 20, 20)  # Sfocatura per ridurre il rumore

        # Correggi la distorsione
        undistorted_image = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y + h, x:x + w]

        # Applica il ritaglio degli angoli
        height, width = undistorted_image.shape[:2]
        undistorted_image = undistorted_image[
            crop_margin:height - crop_margin,  # Ritaglio verticale
            crop_margin:width - crop_margin   # Ritaglio orizzontale
        ]

        # Salva l'immagine ritagliata
        output_path = f"C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/immagini/immagini_drone/immagini_senza_distorsione/img_{i}.jpg"
        print(f"Salvando immagine {output_path}")
        cv2.imwrite(output_path, undistorted_image)

        i += 1
