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


BYTE_PER_FEATURE = 40  # 32 di descrittore ORB piu' due float32 di coordinate


def make_detector(max_features: int = 3000) -> cv2.ORB:
    return cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=6,
        edgeThreshold=15,
        fastThreshold=12,
    )


def feature_ladder(n_frames: int, budget_mb: float = 600.0) -> list[int]:
    """Tetti da provare in successione, dal piu' economico, entro un tetto di memoria.

    `nfeatures` e' un TETTO, non un bersaglio, e quanto debba essere alto non dipende dalla
    camera ma da quanta struttura ripetibile ha il terreno. Su un volo con campi e strade
    tremila feature per scatto bastano e avanzano; su prato raso a 8,6 mm/px sono troppo
    poche perche' nella fascia di sovrapposizione fra due passate -- che vale un decimo del
    fotogramma -- ne cadano abbastanza di ripetibili. Misurato su 23 scatti Altum, contando
    i vincoli superstiti e i cicli indipendenti del grafo:

        tetto  3.000    23 vincoli,  3 fra passate,  2 componenti,   2 cicli
        tetto 10.000    31 vincoli, 10 fra passate,  1 componente,   9 cicli
        tetto 30.000    43 vincoli, 22 fra passate,  1 componente,  21 cicli

    Alzarlo per tutti pero' non si puo': su un volo da 835 scatti trentamila feature sono
    un gigabyte di descrittori, che restano in memoria per tutta la durata (vedi il modulo).
    Da qui la scaletta -- si parte dal valore economico e si sale SOLO se il grafo esce
    malato -- e il tetto di memoria, che accorcia la scaletta da se' sui voli lunghi:
    quello che su venti scatti e' gratis, su ottocento non lo e'.
    """
    scaletta = []
    for tetto in (3000, 12000, 48000):
        if tetto * n_frames * BYTE_PER_FEATURE > budget_mb * 1e6 and scaletta:
            break
        scaletta.append(tetto)
    return scaletta


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

    def set(self, i: int, points: np.ndarray, descriptors: np.ndarray) -> None:
        self.points[i] = points
        self.descriptors[i] = descriptors

    @property
    def nbytes(self) -> int:
        return sum(p.nbytes + d.nbytes for p, d in zip(self.points, self.descriptors))

    def counts(self) -> np.ndarray:
        return np.array([len(p) for p in self.points], dtype=int)
