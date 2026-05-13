"""Assemblaggio del mosaico: bounding box del canvas e composizione first-wins.

Ogni pixel del canvas viene riempito dal PRIMO frame che lo copre, in ordine temporale.
Frame 1 fa la base; ogni frame successivo aggiunge solo le porzioni non ancora coperte.
Niente media, niente peso spaziale, niente blending — l'output e' una progressione pulita
dal primo all'ultimo frame dell'intervallo.
"""
import cv2
import numpy as np


def _warped_corners(M: np.ndarray, w: int, h: int) -> np.ndarray:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(corners, M).reshape(-1, 2)


def compute_canvas(
    transforms: list[np.ndarray], image_sizes: list[tuple[int, int]]
) -> tuple[list[np.ndarray], tuple[int, int], tuple[int, int]]:
    """Trova il bounding box del canvas e trasla le pose."""
    all_corners = [_warped_corners(M, w, h) for M, (w, h) in zip(transforms, image_sizes)]
    pts = np.concatenate(all_corners, axis=0)
    x_min, y_min = np.floor(pts.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(pts.max(axis=0)).astype(int)
    canvas_w, canvas_h = int(x_max - x_min), int(y_max - y_min)
    offset = np.array(
        [[1.0, 0.0, -float(x_min)], [0.0, 1.0, -float(y_min)], [0.0, 0.0, 1.0]]
    )
    shifted = [offset @ M for M in transforms]
    return shifted, (canvas_w, canvas_h), (int(x_min), int(y_min))


def visualize_layout(
    transforms: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    canvas_size: tuple[int, int],
    labels: list[int] | None = None,
) -> np.ndarray:
    """Disegna la geometria di ogni frame nel canvas: rettangolo colorato + numero + freccia "up".

    Il colore segue una progressione HSV in base all'indice temporale (rosso -> giallo -> verde
    -> azzurro -> magenta). Cosi' si vede a colpo d'occhio se i frame sono in sequenza spaziale
    coerente con l'ordine temporale.
    """
    canvas_w, canvas_h = canvas_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    n = len(transforms)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = max(canvas_w, canvas_h) / 1500.0
    thickness = max(2, int(text_scale * 2))

    for i, (M, (w, h)) in enumerate(zip(transforms, image_sizes)):
        hue = int(179.0 * i / max(n - 1, 1))
        color_hsv = np.array([[[hue, 220, 240]]], dtype=np.uint8)
        color = tuple(int(c) for c in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0])

        corners = _warped_corners(M, w, h)
        pts = corners.astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        # Centro immagine e direzione "up" (image-up = -y in pixel) per la freccia
        center_pt = cv2.perspectiveTransform(
            np.array([[[w / 2.0, h / 2.0]]], dtype=np.float64), M
        )[0, 0]
        up_pt = cv2.perspectiveTransform(
            np.array([[[w / 2.0, h / 2.0 - min(w, h) * 0.3]]], dtype=np.float64), M
        )[0, 0]
        cv2.arrowedLine(
            canvas,
            tuple(center_pt.astype(int)),
            tuple(up_pt.astype(int)),
            color=color,
            thickness=thickness,
            tipLength=0.3,
        )

        label = str(labels[i] if labels else (i + 1))
        (tw, th), _ = cv2.getTextSize(label, font, text_scale, thickness)
        org = (int(center_pt[0]) - tw // 2, int(center_pt[1]) + th // 2)
        cv2.putText(canvas, label, org, font, text_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, label, org, font, text_scale, color, thickness, cv2.LINE_AA)

    return canvas


def _strip_mask(
    h: int, w: int, strip_h_px: int, extend_top: bool, extend_bot: bool
) -> np.ndarray:
    """Maschera 8-bit binaria: 255 nella banda centrale, 0 altrove. Banda perpendicolare al volo."""
    half = max(strip_h_px // 2, 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    y_top = 0 if extend_top else max(0, h // 2 - half)
    y_bot = h if extend_bot else min(h, h // 2 + half)
    mask[y_top:y_bot, :] = 255
    return mask


def assemble(
    image_loader,
    transforms: list[np.ndarray],
    canvas_size: tuple[int, int],
    strip_h_px: int,
    progress=None,
) -> tuple[np.ndarray, np.ndarray]:
    """First-wins su striscia centrale: ogni pixel viene dal PRIMO frame che lo copre.

    Niente blending, niente media pesata. Cosi' anche con GPS noise i pixel restano nitidi:
    le linee di giunzione sono visibili come seam ma non c'e' ghosting da disallineamento.

    Ritorna (canvas BGR uint8, maschera bool dei pixel con contenuto).
    """
    canvas_w, canvas_h = canvas_size
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    filled = np.zeros((canvas_h, canvas_w), dtype=bool)
    n = len(transforms)

    iterator = range(n)
    if progress is not None:
        iterator = progress(iterator)

    for i in iterator:
        img = image_loader(i)
        if img is None:
            continue
        h, w = img.shape[:2]
        extend_bot = i == 0
        extend_top = i == n - 1
        mask = _strip_mask(h, w, strip_h_px, extend_top, extend_bot)

        corners = _warped_corners(transforms[i], w, h)
        x0 = max(int(np.floor(corners[:, 0].min())), 0)
        y0 = max(int(np.floor(corners[:, 1].min())), 0)
        x1 = min(int(np.ceil(corners[:, 0].max())), canvas_w)
        y1 = min(int(np.ceil(corners[:, 1].max())), canvas_h)
        if x1 <= x0 or y1 <= y0:
            continue

        T_off = np.array(
            [[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]]
        )
        M_local = T_off @ transforms[i]
        warped = cv2.warpPerspective(
            img, M_local, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR
        )
        warped_mask = cv2.warpPerspective(
            mask, M_local, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST
        )

        valid = (warped.sum(axis=-1) > 0) & (warped_mask > 0)
        filled_roi = filled[y0:y1, x0:x1]
        canvas_roi = canvas[y0:y1, x0:x1]
        new_pixels = valid & ~filled_roi
        canvas_roi[new_pixels] = warped[new_pixels]
        filled_roi[new_pixels] = True

    return canvas, filled
