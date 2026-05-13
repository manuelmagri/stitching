"""Output di debug: immagini con keypoints disegnati.

Funzioni pure: nessun side-effect oltre alla scrittura su `path`.
La pipeline le invoca solo se i flag corrispondenti in `Config.debug` sono attivi.
"""
import cv2


def save_features_image(image_bgr, keypoints, path):
    """Disegna i keypoints sull'immagine BGR e la salva su disco."""
    out = cv2.drawKeypoints(image_bgr, keypoints, None, color=(0, 255, 0))
    cv2.imwrite(str(path), out)
