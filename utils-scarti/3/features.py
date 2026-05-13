"""Feature detection ORB su griglia n×n con budget ripartito."""
import cv2
import numpy as np

from .timing import timed


def divide_into_quadrants(image, n, overlap):
    """Divide l'immagine in n×n quadranti con bordo di sovrapposizione `overlap`.

    Ritorna una lista di tuple (patch, offset_x, offset_y) in ordine row-major.
    Gli offset servono per riportare i keypoint del quadrante al riferimento dell'immagine intera.
    """
    h, w = image.shape[:2]
    step_h = h // n
    step_w = w // n
    quadrants = []
    for i in range(n):
        for j in range(n):
            y1 = max(0, i * step_h - overlap)
            x1 = max(0, j * step_w - overlap)
            y2 = h if i == n - 1 else min(h, (i + 1) * step_h + overlap)
            x2 = w if j == n - 1 else min(w, (j + 1) * step_w + overlap)
            quadrants.append((image[y1:y2, x1:x2], x1, y1))
    return quadrants


class FeatureExtractor:
    """ORB detector con budget feature equamente ripartito sulla griglia n×n.

    L'istanza ORB è costruita una sola volta con `nfeatures = n_features_total // n_grid²`,
    così ogni quadrante contribuisce con lo stesso numero massimo di keypoint senza
    bisogno di mutare lo stato del detector tra una chiamata e l'altra.
    """

    def __init__(self, n_grid, n_features_total, overlap):
        self.n_grid = n_grid
        self.overlap = overlap
        per_quadrant = max(1, n_features_total // (n_grid ** 2))
        self.orb = cv2.ORB_create(nfeatures=per_quadrant)

    @timed
    def detect(self, image_gray):
        """Estrae keypoints e descriptors dall'immagine.

        Ritorna (keypoints, descriptors) con coordinate riportate al riferimento
        dell'immagine originale. `descriptors` è None se nessun keypoint è valido.
        """
        all_keypoints = []
        all_descriptors = []

        for patch, off_x, off_y in divide_into_quadrants(image_gray, self.n_grid, self.overlap):
            kps, des = self.orb.detectAndCompute(patch, None)
            if not kps:
                continue
            for kp in kps:
                kp.pt = (kp.pt[0] + off_x, kp.pt[1] + off_y)
            all_keypoints.extend(kps)
            if des is not None:
                all_descriptors.append(des)

        descriptors = np.vstack(all_descriptors) if all_descriptors else None
        return all_keypoints, descriptors
