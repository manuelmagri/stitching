"""Mosaic renderer "best-pixel" tipo Voronoi su canvas UTM-locale.

Per ogni pixel del canvas vince il frame con lo score warpato massimo, dove
lo score sorgente di ogni frame e' la distanza euclidea dal bordo dell'immagine
(0 ai bordi, massima al centro). Risultato:

  - niente blending -> niente ghosting su oggetti 3D (alberi, edifici);
  - niente seam rettangolari -> le cuciture cadono naturalmente lungo le
    bisettrici fra i centri dei frame, dove la qualita' di entrambi i contributi
    e' simile;
  - placement metrico esatto via omografia analitica (compatibile con
    `homography_curve.omografia_immagine_su_piano`).

API:
  crea_canvas(homographies, img_w, img_h, res_m_per_px)  ->  MosaicCanvas
  aggiungi_immagine(canvas, image, H_img_to_world, tile_u_px, tile_v_px, cx_px, cy_px)
  salva_jpg(canvas, output_path)
  diagnostica_seam_residui(canvas, image_paths, homographies, indices)
    ritorna un breve report sui giunti tra coppie consecutive di frame
    warpati: distanza media in pixel del canvas tra match ORB inlier delle
    due proiezioni. Utile per decidere se serve un refinement geometrico.

Compatibilita': `tile_u_px / tile_v_px / cx_px / cy_px` sono accettati ma non
usati in questa Fase 1 (il best-pixel non ha bisogno di un tile rettangolare).
Restano nella firma per non rompere `main.py`: una Fase 2 potra' riutilizzarli
per "premiare" il centro tile rispetto al centro ottico.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np


# Soglia di sicurezza: oltre questo numero di pixel sul canvas, alziamo un errore
# anziche' allocare gigabyte di RAM per un mosaico patologico.
_MAX_CANVAS_PIXELS = 400_000_000


@dataclass
class MosaicCanvas:
    """Canvas mondo-allineato: ogni pixel ha coord UTM-locale ricavabili da `bounds_world` + `res_m_per_px`."""
    image: np.ndarray              # (H, W, 3) uint8 BGR
    best_score: np.ndarray         # (H, W) float32 - distanza dal bordo del frame "vincente"
    res_m_per_px: float            # risoluzione metrica del canvas
    bounds_world: tuple[float, float, float, float]
    # bounds_world = (xmin, ymin, xmax, ymax) in coord mondo (UTM-locale).
    # Per convenzione del writer GeoTIFF: pixel (0, 0) corrisponde a (xmin, ymax).


# ---------------------------------------------------------------------------
# Helper interni
# ---------------------------------------------------------------------------

def _proietta_angoli(H: np.ndarray, w: int, h: int) -> np.ndarray:
    """Proietta i 4 angoli (u, v) dell'immagine in coord mondo via H 3x3."""
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
    P = H @ corners                  # (3, 4)
    P /= P[2:3, :]
    return P[:2, :].T                # (4, 2)


def _T_world_to_canvas(xmin: float, ymax: float, res: float) -> np.ndarray:
    """3x3 affine che porta (x_world, y_world, 1) -> (col_canvas, row_canvas, 1)."""
    return np.array([
        [1.0 / res, 0.0, -xmin / res],
        [0.0, -1.0 / res, ymax / res],
        [0.0,  0.0,        1.0],
    ], dtype=np.float64)


def _score_sorgente(h: int, w: int) -> np.ndarray:
    """Distance transform di una mask piena tranne 1 pixel sui bordi.

    Risultato: 0 sui bordi del frame, ~ min(h, w)/2 al centro. Da usare come
    "qualita' del pixel sorgente": il pixel piu' vicino al bordo perde
    contro un pixel piu' centrale di un frame vicino.
    """
    mask = np.ones((h, w), dtype=np.uint8)
    mask[:1, :] = 0
    mask[-1:, :] = 0
    mask[:, :1] = 0
    mask[:, -1:] = 0
    return cv2.distanceTransform(mask, cv2.DIST_L2, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def crea_canvas(homographies: list[np.ndarray],
                img_w: int, img_h: int,
                res_m_per_px: float) -> MosaicCanvas:
    """Costruisce un canvas vuoto dimensionato sulla bbox di tutti i frame proiettati.

    Le `homographies` sono 3x3 immagine -> mondo (UTM-locale). La bbox e' la
    bounding box delle proiezioni dei 4 angoli di ogni frame, espansa di 1 px
    per evitare clipping ai bordi.
    """
    if not homographies:
        raise ValueError("Lista omografie vuota.")
    res = float(res_m_per_px)
    if res <= 0:
        raise ValueError(f"res_m_per_px deve essere > 0, ricevuto {res}.")

    all_pts = np.vstack([_proietta_angoli(H, img_w, img_h) for H in homographies])
    xmin = float(all_pts[:, 0].min())
    xmax = float(all_pts[:, 0].max())
    ymin = float(all_pts[:, 1].min())
    ymax = float(all_pts[:, 1].max())

    W = int(np.ceil((xmax - xmin) / res)) + 2
    H = int(np.ceil((ymax - ymin) / res)) + 2
    if W <= 0 or H <= 0:
        raise RuntimeError(f"Canvas degenere: W={W}, H={H}.")
    if W * H > _MAX_CANVAS_PIXELS:
        raise RuntimeError(
            f"Canvas troppo grande ({W}x{H} = {W*H:.2e} px > {_MAX_CANVAS_PIXELS:.0e}). "
            "Alza res_m_per_px o riduci il volo."
        )

    image = np.zeros((H, W, 3), dtype=np.uint8)
    best_score = np.full((H, W), -1.0, dtype=np.float32)
    return MosaicCanvas(
        image=image,
        best_score=best_score,
        res_m_per_px=res,
        bounds_world=(xmin, ymin, xmax, ymax),
    )


def aggiungi_immagine(canvas: MosaicCanvas,
                      image: np.ndarray,
                      H_img_to_world: np.ndarray,
                      tile_u_px: int | None = None,
                      tile_v_px: int | None = None,
                      cx_px: float | None = None,
                      cy_px: float | None = None) -> None:
    """Compone `image` sul canvas via warp + best-pixel-score.

    Aggiorna `canvas.image` e `canvas.best_score` in-place: ogni pixel del
    canvas viene sovrascritto solo se il punteggio (distance-from-edge)
    dello score warpato e' strettamente maggiore di quello gia' presente.
    Parametri `tile_u_px / tile_v_px / cx_px / cy_px` accettati per
    compatibilita' di firma ma non usati in questa Fase 1.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Atteso BGR (H, W, 3), ricevuto shape {image.shape}.")
    h_src, w_src = image.shape[:2]

    xmin, ymin, xmax, ymax = canvas.bounds_world
    T = _T_world_to_canvas(xmin, ymax, canvas.res_m_per_px)
    M_img_to_canvas = (T @ np.asarray(H_img_to_world, dtype=np.float64)).astype(np.float64)

    H_cv, W_cv = canvas.image.shape[:2]
    warped_img = cv2.warpPerspective(
        image, M_img_to_canvas, (W_cv, H_cv),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # NEAREST per lo score: evita "aloni" a valori bassi attorno ai bordi
    # warpati che farebbero competere pixel "appena fuori" frame con quelli
    # interni di un frame vicino.
    score_src = _score_sorgente(h_src, w_src)
    warped_score = cv2.warpPerspective(
        score_src, M_img_to_canvas, (W_cv, H_cv),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    update = warped_score > canvas.best_score
    np.copyto(canvas.best_score, warped_score, where=update)
    np.copyto(canvas.image, warped_img, where=update[..., None])


def salva_jpg(canvas: MosaicCanvas, output_path: str, quality: int = 92) -> str:
    """Salva `canvas.image` come JPG. Crea le cartelle se mancano."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    ok = cv2.imwrite(output_path, canvas.image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError(f"Salvataggio JPG fallito: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Diagnostica seam residual
# ---------------------------------------------------------------------------

def diagnostica_seam_residui(canvas: MosaicCanvas,
                             image_paths: list[str],
                             homographies: list[np.ndarray],
                             indices: list[int],
                             *,
                             n_features: int = 1500,
                             min_inliers: int = 20,
                             scale: float = 1.0) -> dict | None:
    """Misura lo scollamento residuo tra frame consecutivi nel canvas.

    Per ogni coppia (i, j) consecutiva in `indices`:
      1. Carica le due immagini, fa ORB+BFMatcher cross-check
      2. Stima estimateAffinePartial2D RANSAC fra i punti
      3. Mappa gli inlier dal frame sorgente al canvas via le rispettive H
      4. Misura la distanza euclidea tra le due proiezioni canvas

    Se la geometria fosse perfetta i punti caderebbero esattamente sullo
    stesso pixel canvas; in pratica c'e' qualche pixel di scollamento per
    errori GPS / yaw / AGL residui. Output utile per decidere se vale la
    pena fare un refinement (Fase 2).
    """
    if len(indices) < 2:
        return None

    xmin, ymin, xmax, ymax = canvas.bounds_world
    T = _T_world_to_canvas(xmin, ymax, canvas.res_m_per_px)

    orb = cv2.ORB_create(nfeatures=n_features)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    all_diff: list[np.ndarray] = []
    n_ok = 0
    n_failed = 0

    for k in range(len(indices) - 1):
        i, j = indices[k], indices[k + 1]
        img_a = cv2.imread(image_paths[i])
        img_b = cv2.imread(image_paths[j])
        if img_a is None or img_b is None:
            n_failed += 1
            continue
        if scale != 1.0:
            img_a = cv2.resize(img_a, (int(img_a.shape[1] * scale), int(img_a.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
            img_b = cv2.resize(img_b, (int(img_b.shape[1] * scale), int(img_b.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        gA = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gB = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
        kpA, deA = orb.detectAndCompute(gA, None)
        kpB, deB = orb.detectAndCompute(gB, None)
        if deA is None or deB is None or len(deA) < min_inliers or len(deB) < min_inliers:
            n_failed += 1
            continue
        matches = bf.match(deA, deB)
        if len(matches) < min_inliers:
            n_failed += 1
            continue
        ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])
        _, inliers = cv2.estimateAffinePartial2D(
            ptsA, ptsB, method=cv2.RANSAC, ransacReprojThreshold=3.0,
        )
        if inliers is None or int(inliers.sum()) < min_inliers:
            n_failed += 1
            continue
        inl = inliers.ravel().astype(bool)
        ptsA_inl = np.hstack([ptsA[inl], np.ones((int(inl.sum()), 1), dtype=np.float32)])
        ptsB_inl = np.hstack([ptsB[inl], np.ones((int(inl.sum()), 1), dtype=np.float32)])
        # Mappe pixel_immagine -> pixel_canvas
        M_i = (T @ np.asarray(homographies[i], dtype=np.float64))
        M_j = (T @ np.asarray(homographies[j], dtype=np.float64))
        # Applico le 3x3 perspectice: (col, row, 1) -> (col*, row*, w); divido per w.
        pA = ptsA_inl @ M_i.T
        pA = pA[:, :2] / pA[:, 2:3]
        pB = ptsB_inl @ M_j.T
        pB = pB[:, :2] / pB[:, 2:3]
        all_diff.append(np.linalg.norm(pA - pB, axis=1))
        n_ok += 1

    if not all_diff:
        return {"n_ok": 0, "n_failed": n_failed, "median_px": None, "p90_px": None, "max_px": None}
    diffs = np.concatenate(all_diff)
    return {
        "n_ok": n_ok,
        "n_failed": n_failed,
        "median_px": float(np.median(diffs)),
        "p90_px": float(np.percentile(diffs, 90)),
        "max_px": float(diffs.max()),
    }
