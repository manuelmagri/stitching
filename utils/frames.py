"""Accesso alle immagini: lettura pigra, finestra scorrevole, due risoluzioni.

Il volo non entra in memoria e non deve provarci: 835 scatti a piena risoluzione sono
52 GB di pixel. Qui si tiene aperta una finestra di pochi fotogrammi alla volta, si legge
solo cio' che serve nel momento in cui serve, e il piu' vecchio esce appena la finestra e'
piena. Cio' che sopravvive al passaggio non sono i pixel ma i descrittori, che per tutto
il volo pesano 150 MB (vedi `utils.features`).

La riduzione usa i flag IMREAD_REDUCED di OpenCV, che decodificano direttamente alla
risoluzione ridotta invece di decodificare tutto e poi ridimensionare: a un quarto e'
circa otto volte piu' veloce, ed e' gratis perche' avviene dentro il decoder JPEG.

Attenzione al fattore di scala: la riduzione arrotonda per eccesso, quindi 4909 px a un
quarto diventano 1228 e non 1227,25 -- il fattore vero e' 3,99756, non 4. `scale` lo
espone calcolato sulle dimensioni reali, ed e' quello che `utils.poses.rescale_poses`
deve ricevere per riportare le pose a piena risoluzione.
"""
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

_FLAGS = {
    1: cv2.IMREAD_COLOR,
    2: cv2.IMREAD_REDUCED_COLOR_2,
    4: cv2.IMREAD_REDUCED_COLOR_4,
    8: cv2.IMREAD_REDUCED_COLOR_8,
}

# Lato lungo minimo a cui stimare le pose. E' l'unica costante di taratura rimasta nella
# pipeline, e va letta per quello che e': un valore MISURATO su due voli, non dedotto.
# Contando i vincoli che sopravvivono al matching:
#
#     DJI    (4909 px)   riduzione 4 -> 1228 px, 37,0 mm/px   96 vincoli su 123 coppie
#     Altum  (1591 px)   riduzione 1 -> 1591 px,  4,3 mm/px   19 vincoli,  0 fra passate
#                        riduzione 2 ->  795 px,  8,6 mm/px   23 vincoli,  3 fra passate
#                        riduzione 4 ->  397 px, 17,3 mm/px   23 vincoli,  3 fra passate
#
# A piena risoluzione l'erba a 4,3 mm/px e' rumore: i descrittori si agganciano a niente, e
# fra le passate non passa un solo vincolo. Ridurre la media via e lascia le strutture
# grandi. Perche' entrambi i voli scelgano la riduzione giusta il valore deve stare fra 614
# e 795 -- una finestra larga il 25% -- e 700 ne sta in mezzo.
#
# Non e' una legge fisica: la scala a cui un terreno ha struttura dipende dal terreno, e su
# un volo molto diverso questo numero puo' sbagliare. Il sintomo pero' e' immediato, ed e'
# gia' stampato: componenti e cicli indipendenti sui vincoli SUPERSTITI. Un grafo che si
# spezza mentre le impronte si sovrappongono e' quasi sempre questo.
LATO_LUNGO_MINIMO = 700


def reduction_for(image_size: tuple[int, int], min_long_side: int = LATO_LUNGO_MINIMO) -> int:
    """La riduzione piu' spinta che lascia il lato lungo sopra `min_long_side`.

    Si fissa la RISOLUZIONE DI LAVORO e non il divisore, che tarato a 4 varrebbe per una
    camera sola: e' la stessa parametrizzazione che `compositing.plan` usa con
    `seam_megapix`.
    """
    lungo = max(image_size)
    for riduzione in sorted(_FLAGS, reverse=True):
        if lungo / riduzione >= min_long_side:
            return riduzione
    return 1


class FrameReader:
    """Lettore a finestra scorrevole su una lista di percorsi.

    `cache_size` e' l'ampiezza della finestra. Il valore di default tiene la coppia su cui
    si sta lavorando, che e' quanto basta per scorrere il volo in ordine; alzarlo aiuta
    solo se si salta avanti e indietro.
    """

    def __init__(
        self,
        paths: list[Path],
        full_size: tuple[int, int],
        reduce: int = 1,
        cache_size: int = 2,
    ):
        if reduce not in _FLAGS:
            raise ValueError(f"riduzione non supportata: {reduce}. Usa 1, 2, 4 o 8.")
        if cache_size < 1:
            raise ValueError("cache_size deve essere almeno 1")

        self.paths = list(paths)
        self.full_size = full_size
        self.reduce = reduce
        self.cache_size = cache_size
        self._flag = _FLAGS[reduce]
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._size: tuple[int, int] | None = None
        self.letture = 0

    @property
    def image_size(self) -> tuple[int, int]:
        """(larghezza, altezza) dopo la riduzione. Legge un fotogramma se serve."""
        if self._size is None:
            if not self.paths:
                raise RuntimeError("nessun fotogramma da leggere")
            self.get(0)
        return self._size

    @property
    def scale(self) -> float:
        """Fattore fra piena risoluzione e risoluzione ridotta, dalle dimensioni reali."""
        w, h = self.image_size
        sx = self.full_size[0] / w
        sy = self.full_size[1] / h
        if abs(sx - sy) / max(sx, sy) > 1e-3:
            raise RuntimeError(
                f"riduzione anisotropa: {sx:.5f} in x contro {sy:.5f} in y. "
                "Le pose sono similarita' e non possono assorbire una scala diversa per asse."
            )
        return 0.5 * (sx + sy)

    def get(self, i: int) -> np.ndarray | None:
        """Fotogramma `i` in BGR, o None se illeggibile. Lo scarto e' il meno recente."""
        cached = self._cache.get(i)
        if cached is not None:
            self._cache.move_to_end(i)
            return cached

        img = cv2.imread(str(self.paths[i]), self._flag)
        self.letture += 1
        if img is None:
            return None

        if self._size is None:
            self._size = (img.shape[1], img.shape[0])

        self._cache[i] = img
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return img

    def clear(self) -> None:
        self._cache.clear()
