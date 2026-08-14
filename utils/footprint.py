"""Impronta a terra di uno scatto, e quanto due impronte si sovrappongono.

E' il cuore dell'idea: sapendo posizione, rotta e quota si sa in anticipo che cosa copre
ogni scatto, quindi si tentano solo gli accoppiamenti geometricamente plausibili invece
di provarli tutti.

L'impronta viene derivata dalla POSA, non ricalcolata da posizione e rotta: cosi' non puo'
sfasarsi rispetto alle convenzioni di segno usate altrove, perche' e' letteralmente il
rettangolo dell'immagine trasformato dalla posa. Per la stessa ragione si lavora in pixel
canvas e non in metri -- la sovrapposizione e' una frazione, quindi le unita' si
semplificano.

Vale finche' la camera guarda il nadir e il terreno e' pianeggiante; `utils.flight` esclude
gia' gli scatti in cui il gimbal non era al nadir.
"""
import numpy as np

import cv2


def quad_from_pose(M: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """I quattro angoli dell'immagine portati sul canvas dalla posa `M`, array (4, 2)."""
    w, h = image_size
    corners = np.array(
        [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]], dtype=np.float64
    )
    return (corners @ M.T)[:, :2]


def quads_from_poses(
    transforms: list[np.ndarray], image_size: tuple[int, int]
) -> np.ndarray:
    """Array (N, 4, 2) con l'impronta di ogni posa."""
    return np.array([quad_from_pose(M, image_size) for M in transforms], dtype=np.float64)


def polygon_area(quad: np.ndarray) -> float:
    """Area di un poligono convesso con la formula del laccio delle scarpe."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def overlap_fraction(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Frazione di sovrapposizione fra due impronte, in [0, 1].

    Il denominatore e' l'area della PIU' PICCOLA delle due, quindi la domanda a cui si
    risponde e' "almeno una delle due immagini e' coperta dall'altra per questa frazione?".
    Con quote simili le due aree coincidono e la scelta e' indifferente; conta solo se il
    volo cambia quota in modo marcato.
    """
    area, _ = cv2.intersectConvexConvex(
        quad_a.astype(np.float32), quad_b.astype(np.float32)
    )
    if area <= 0.0:
        return 0.0
    riferimento = min(polygon_area(quad_a), polygon_area(quad_b))
    if riferimento <= 0.0:
        return 0.0
    return float(area) / riferimento


def centers_and_radii(quads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centro e raggio circoscritto di ogni impronta, per il prefiltro sulle distanze.

    Due impronte i cui centri distano piu' della somma dei raggi non possono toccarsi:
    scartarle con un confronto di distanze evita l'intersezione esatta sulla stragrande
    maggioranza delle coppie.
    """
    centers = quads.mean(axis=1)
    radii = np.linalg.norm(quads - centers[:, None, :], axis=2).max(axis=1)
    return centers, radii


def footprint_size_m(gsd_m_per_px: float, image_size: tuple[int, int]) -> tuple[float, float]:
    """Impronta in metri (larghezza cross-track, altezza along-track), per diagnostica."""
    w, h = image_size
    return w * gsd_m_per_px, h * gsd_m_per_px
