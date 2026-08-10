"""Geometria del canvas: dove va a finire ogni scatto, e quanto e' grande il risultato.

Qui si stabilisce solo la collocazione. Come i pixel sovrapposti vengano combinati e'
affare di `utils.compositing`, che applica la successione classica guadagni / cuciture /
fusione multibanda.

Il canvas a piena risoluzione del volo di prova e' 37188 x 38163 pixel, cioe' 1,4
gigapixel: allocarlo tutto non e' una strada percorribile, quindi la composizione e'
guidata dall'USCITA e non dall'ingresso. Si scorre il canvas per bande orizzontali e per
ciascuna si caricano solo gli scatti che la intersecano; il costo e' qualche rilettura, in
cambio il picco di memoria e' quello di una banda sola e le bande si scrivono man mano nel
GeoTIFF, che e' gia' organizzato a blocchi.
"""
import numpy as np

import cv2


def _warped_corners(M: np.ndarray, w: int, h: int) -> np.ndarray:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, M).reshape(-1, 2)


def compute_canvas(
    transforms: list[np.ndarray], image_size: tuple[int, int]
) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int]]:
    """Riquadro che contiene tutte le impronte, e pose traslate perche' parta da (0, 0).

    Ritorna (pose traslate, (larghezza, altezza), (offset_x, offset_y)). L'offset serve a
    ritrovare l'origine geografica: il pixel (0, 0) del canvas finale corrispondeva al
    pixel (offset_x, offset_y) del sistema in cui le pose erano espresse.
    """
    w, h = image_size
    angoli = np.concatenate([_warped_corners(M, w, h) for M in transforms], axis=0)
    x_min, y_min = np.floor(angoli.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(angoli.max(axis=0)).astype(int)

    offset = np.array(
        [[1.0, 0.0, -float(x_min)], [0.0, 1.0, -float(y_min)], [0.0, 0.0, 1.0]]
    )
    return (
        [offset @ M for M in transforms],
        (int(x_max - x_min), int(y_max - y_min)),
        (int(x_min), int(y_min)),
    )


def frame_boxes(
    transforms: list[np.ndarray], image_size: tuple[int, int], canvas_size: tuple[int, int]
) -> list[tuple[int, int, int, int]]:
    """Riquadro (x0, y0, x1, y1) di ogni scatto sul canvas, ritagliato ai bordi."""
    w, h = image_size
    canvas_w, canvas_h = canvas_size
    riquadri = []
    for M in transforms:
        angoli = _warped_corners(M, w, h)
        x0 = max(int(np.floor(angoli[:, 0].min())), 0)
        y0 = max(int(np.floor(angoli[:, 1].min())), 0)
        x1 = min(int(np.ceil(angoli[:, 0].max())), canvas_w)
        y1 = min(int(np.ceil(angoli[:, 1].max())), canvas_h)
        riquadri.append((x0, y0, x1, y1))
    return riquadri


def visualize_layout(
    transforms: list[np.ndarray],
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    labels: list[int] | None = None,
    max_side: int = 2000,
) -> np.ndarray:
    """Disegno diagnostico delle impronte: rettangolo colorato, numero e freccia di rotta.

    Il colore segue l'ordine temporale, da rosso a magenta, cosi' si vede a colpo d'occhio
    se le pose sono coerenti con la sequenza di volo. Va guardato PRIMA di comporre il
    mosaico: costa un istante e mostra subito una passata fuori posto, mentre accorgersene
    dal mosaico costa l'intera composizione.
    """
    canvas_w, canvas_h = canvas_size
    fattore = min(1.0, max_side / max(canvas_w, canvas_h))
    W, H = max(int(canvas_w * fattore), 1), max(int(canvas_h * fattore), 1)

    tela = np.zeros((H, W, 3), dtype=np.uint8)
    w, h = image_size
    n = len(transforms)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scala_testo = max(W, H) / 1500.0
    spessore = max(1, int(scala_testo * 2))
    S = np.diag([fattore, fattore, 1.0])

    for i, M in enumerate(transforms):
        tono = int(179.0 * i / max(n - 1, 1))
        colore = tuple(
            int(c)
            for c in cv2.cvtColor(np.uint8([[[tono, 220, 240]]]), cv2.COLOR_HSV2BGR)[0, 0]
        )
        Ms = S @ M

        cv2.polylines(
            tela, [_warped_corners(Ms, w, h).astype(np.int32)], True, colore, spessore
        )
        centro = cv2.perspectiveTransform(
            np.array([[[w / 2.0, h / 2.0]]], dtype=np.float64), Ms
        )[0, 0]
        avanti = cv2.perspectiveTransform(
            np.array([[[w / 2.0, h / 2.0 - min(w, h) * 0.3]]], dtype=np.float64), Ms
        )[0, 0]
        cv2.arrowedLine(
            tela, tuple(centro.astype(int)), tuple(avanti.astype(int)), colore, spessore, tipLength=0.3
        )

        testo = str(labels[i] if labels else (i + 1))
        (tw, th), _ = cv2.getTextSize(testo, font, scala_testo, spessore)
        org = (int(centro[0]) - tw // 2, int(centro[1]) + th // 2)
        cv2.putText(tela, testo, org, font, scala_testo, (255, 255, 255), spessore + 2, cv2.LINE_AA)
        cv2.putText(tela, testo, org, font, scala_testo, colore, spessore, cv2.LINE_AA)

    return tela
