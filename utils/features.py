"""Feature detection con ORB su frame in scala di grigi."""
import cv2
import numpy as np


def make_detector(max_features: int = 1500) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=15,
        fastThreshold=12,
    )


def detect(detector: cv2.ORB, image_gray: np.ndarray):
    """Restituisce (keypoints, descriptors). descriptors puo' essere None."""
    return detector.detectAndCompute(image_gray, None)
