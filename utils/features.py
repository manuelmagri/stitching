"""Rilevamento ORB e conservazione dei descrittori per tutto il volo.

E' qui che si vede perche' la finestra scorrevole di `utils.frames` non impedisce di
agganciare passate lontane nel tempo: i pixel di uno scatto durano il tempo di estrarne le
feature, i descrittori restano. Un fotogramma a piena risoluzione occupa 52 MB; i suoi
3000 descrittori ORB, con le coordinate, ne occupano 0,18. L'intero volo sta in 150 MB,
quindi non c'e' ragione di buttarne via nessuno.

I keypoint vengono ridotti subito alle sole coordinate. Tenere gli oggetti cv2.KeyPoint
significherebbe due milioni e mezzo di oggetti Python, e di quegli oggetti alla stima
della similarita' serve solo `.pt`.
"""
import cv2
import numpy as np


def make_detector(max_features: int = 3000) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=15,
        fastThreshold=12,
    )


def detect(detector: cv2.ORB, image_bgr: np.ndarray):
    """(punti (N,2) float32, descrittori (N,32) uint8). Entrambi vuoti se non trova nulla."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    if not keypoints or descriptors is None:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 32), dtype=np.uint8)
    punti = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    return punti, descriptors


class DescriptorStore:
    """Punti e descrittori di ogni scatto, indicizzati per posizione."""

    def __init__(self, n: int):
        self.points: list[np.ndarray] = [np.empty((0, 2), dtype=np.float32)] * n
        self.descriptors: list[np.ndarray] = [np.empty((0, 32), dtype=np.uint8)] * n

    def __len__(self) -> int:
        return len(self.points)

    def set(self, i: int, points: np.ndarray, descriptors: np.ndarray) -> None:
        self.points[i] = points
        self.descriptors[i] = descriptors

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes + d.nbytes for p, d in zip(self.points, self.descriptors))

    def counts(self) -> np.ndarray:
        return np.array([len(p) for p in self.points], dtype=int)
