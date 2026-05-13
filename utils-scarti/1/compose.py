"""Composizione dei chunk-passata su un canvas UTM globale.

Strategia:
    1. Calcola la bounding box UTM globale unendo tutte le bounding box dei chunk.
    2. Alloca un canvas alla risoluzione `canvas_gsd_m_per_px`.
    3. Copia ogni chunk nel canvas globale alla sua posizione UTM (semplice copia
       di pixel: i chunk vivono gia' nella stessa griglia UTM).
    4. Average blending nelle zone di sovrapposizione tra passate adiacenti.

Stub di refinement GPS-aware: dopo la composizione iniziale, e' possibile
trovare match di feature nelle zone di sovrapposizione tra chunk adiacenti
(GPS-aware: si confrontano solo coppie di passate distanti < raggio fissato)
e applicare una piccola correzione (traslazione 2D) per allineare i chunk.
La correzione viene fatta sulle posizioni UTM dei chunk, mantenendo invariata
la loro grandezza/rotazione.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .features import detect_orb_grid
from .matching import (
    homography_ransac,
    make_flann,
    match_descriptors,
    matched_points,
)
from .stitch_chunk import Chunk


@dataclass
class GlobalMosaic:
    image: np.ndarray            # BGR uint8
    coverage: np.ndarray         # uint8
    min_E: float
    max_N: float
    gsd_m_per_px: float


def compose_chunks(
    chunks: list[Chunk],
    *,
    canvas_gsd_m_per_px: float | None = None,
) -> GlobalMosaic:
    """Compone i chunk in un canvas UTM globale con average blending."""
    if not chunks:
        raise ValueError("Nessun chunk da comporre")

    g = float(canvas_gsd_m_per_px or chunks[0].gsd_m_per_px)

    min_E = min(c.min_E for c in chunks)
    max_N = max(c.max_N for c in chunks)
    max_E = max(c.min_E + c.image.shape[1] * c.gsd_m_per_px for c in chunks)
    min_N = min(c.max_N - c.image.shape[0] * c.gsd_m_per_px for c in chunks)

    W = int(np.ceil((max_E - min_E) / g)) + 2
    H = int(np.ceil((max_N - min_N) / g)) + 2

    canvas = np.zeros((H, W, 3), dtype=np.uint32)
    counts = np.zeros((H, W), dtype=np.uint16)

    for c in chunks:
        # Punto top-left del chunk in pixel del canvas globale:
        col0 = int(round((c.min_E - min_E) / g))
        row0 = int(round((max_N - c.max_N) / g))

        ch_h, ch_w = c.image.shape[:2]
        # Riscala il chunk se la sua GSD differisce dal canvas globale
        if abs(c.gsd_m_per_px - g) > 1e-9:
            scale = c.gsd_m_per_px / g
            new_w = int(round(ch_w * scale))
            new_h = int(round(ch_h * scale))
            chunk_img = cv2.resize(c.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            chunk_img = c.image

        ch_h, ch_w = chunk_img.shape[:2]
        # Clamp ai limiti del canvas
        r1 = max(0, row0)
        c1 = max(0, col0)
        r2 = min(H, row0 + ch_h)
        c2 = min(W, col0 + ch_w)
        if r2 <= r1 or c2 <= c1:
            continue

        sub = chunk_img[r1 - row0: r2 - row0, c1 - col0: c2 - col0]
        mask = (sub.sum(axis=2) > 0).astype(np.uint16)
        canvas[r1:r2, c1:c2] += sub.astype(np.uint32) * mask[..., None]
        counts[r1:r2, c1:c2] += mask

    counts_safe = np.maximum(counts, 1).astype(np.uint32)
    out = (canvas // counts_safe[..., None]).astype(np.uint8)
    coverage = np.minimum(counts.astype(np.uint16) * 32, 255).astype(np.uint8)

    return GlobalMosaic(
        image=out, coverage=coverage,
        min_E=float(min_E), max_N=float(max_N),
        gsd_m_per_px=float(g),
    )


# ---------------------------------------------------------------------------
# Refinement opzionale: match GPS-aware tra passate adiacenti
# ---------------------------------------------------------------------------

def _chunk_overlap_aabb(a: Chunk, b: Chunk) -> tuple[float, float, float, float] | None:
    """Bounding box (min_E, min_N, max_E, max_N) dell'intersezione tra due chunk in UTM."""
    a_min_E = a.min_E
    a_max_E = a.min_E + a.image.shape[1] * a.gsd_m_per_px
    a_max_N = a.max_N
    a_min_N = a.max_N - a.image.shape[0] * a.gsd_m_per_px

    b_min_E = b.min_E
    b_max_E = b.min_E + b.image.shape[1] * b.gsd_m_per_px
    b_max_N = b.max_N
    b_min_N = b.max_N - b.image.shape[0] * b.gsd_m_per_px

    min_E = max(a_min_E, b_min_E)
    max_E = min(a_max_E, b_max_E)
    min_N = max(a_min_N, b_min_N)
    max_N = min(a_max_N, b_max_N)
    if max_E <= min_E or max_N <= min_N:
        return None
    return min_E, min_N, max_E, max_N


def _crop_chunk_to_aabb(c: Chunk, aabb: tuple[float, float, float, float]) -> tuple[np.ndarray, int, int]:
    """Ritaglia il chunk all'AABB UTM. Ritorna (patch, off_row, off_col) nel chunk."""
    min_E, min_N, max_E, max_N = aabb
    g = c.gsd_m_per_px
    col0 = int(np.floor((min_E - c.min_E) / g))
    col1 = int(np.ceil((max_E - c.min_E) / g))
    row0 = int(np.floor((c.max_N - max_N) / g))
    row1 = int(np.ceil((c.max_N - min_N) / g))
    col0 = max(0, col0); col1 = min(c.image.shape[1], col1)
    row0 = max(0, row0); row1 = min(c.image.shape[0], row1)
    return c.image[row0:row1, col0:col1].copy(), row0, col0


def refine_chunks_inter_pass(
    chunks: list[Chunk],
    *,
    n_features_total: int = 3000,
    n_grid: int = 4,
    lowe_ratio: float = 0.7,
    ransac_thresh: float = 3.0,
    min_inliers: int = 20,
    verbose: bool = True,
) -> list[Chunk]:
    """Stima una traslazione 2D di correzione per ciascun chunk, basata su match
    nelle zone di sovrapposizione tra passate adiacenti.

    Il primo chunk e' fissato come riferimento; gli altri vengono spostati con
    una traslazione media derivata dai match con i chunk gia' "ancorati".

    Approccio semplificato: stimiamo solo Tx, Ty in metri (no rotazione/scala),
    perche' la rotazione gia' e' giusta (chunks costruiti in UTM).
    """
    if len(chunks) <= 1:
        return chunks

    flann = make_flann()
    refined: list[Chunk] = [chunks[0]]
    for k in range(1, len(chunks)):
        c_b = chunks[k]
        # Cerca un chunk gia' ancorato che si sovrappone a c_b
        best_dE: list[float] = []
        best_dN: list[float] = []
        for c_a in refined:
            aabb = _chunk_overlap_aabb(c_a, c_b)
            if aabb is None:
                continue
            patch_a, ra0, ca0 = _crop_chunk_to_aabb(c_a, aabb)
            patch_b, rb0, cb0 = _crop_chunk_to_aabb(c_b, aabb)
            if patch_a.size == 0 or patch_b.size == 0:
                continue
            ga = cv2.cvtColor(patch_a, cv2.COLOR_BGR2GRAY)
            gb = cv2.cvtColor(patch_b, cv2.COLOR_BGR2GRAY)
            kpa, dea = detect_orb_grid(ga, n_grid, n_features_total)
            kpb, deb = detect_orb_grid(gb, n_grid, n_features_total)
            good = match_descriptors(flann, dea, deb, lowe_ratio=lowe_ratio, min_matches=min_inliers)
            if not good:
                continue
            qa, qb = matched_points(kpa, kpb, good)
            H, mask = homography_ransac(qa, qb, ransac_thresh=ransac_thresh)
            if mask is None or int(mask.sum()) < min_inliers:
                continue
            qa = qa[mask]; qb = qb[mask]
            # Differenza media (in pixel del crop)
            dxy_px = (qa - qb).mean(axis=0)
            # Patch_a vive dentro c_a ad offset (ra0, ca0). qa = (col, row) nel patch_a.
            # Posizione di un punto in UTM (per chunk c_a) = (c_a.min_E + (ca0 + col_a)*g, c_a.max_N - (ra0 + row_a)*g)
            # Differenza UTM tra il punto in c_a e il punto in c_b:
            #   dE = (c_a.min_E + (ca0 + col_a) * g) - (c_b.min_E + (cb0 + col_b) * g)
            #   dN = (c_b.max_N - (rb0 + row_b) * g) - (c_a.max_N - (ra0 + row_a) * g)
            # Possiamo ricavare lo shift UTM di c_b per allinearsi a c_a:
            mean_a = qa.mean(axis=0)  # (col, row) nel patch_a
            mean_b = qb.mean(axis=0)
            E_a = c_a.min_E + (ca0 + mean_a[0]) * c_a.gsd_m_per_px
            N_a = c_a.max_N - (ra0 + mean_a[1]) * c_a.gsd_m_per_px
            E_b = c_b.min_E + (cb0 + mean_b[0]) * c_b.gsd_m_per_px
            N_b = c_b.max_N - (rb0 + mean_b[1]) * c_b.gsd_m_per_px
            dE = E_a - E_b
            dN = N_a - N_b
            best_dE.append(dE)
            best_dN.append(dN)

        if not best_dE:
            refined.append(c_b)
            continue
        dE = float(np.mean(best_dE))
        dN = float(np.mean(best_dN))
        if verbose:
            print(f"[compose] refine chunk pass={c_b.pass_index}: shift UTM (dE={dE:+.2f} m, dN={dN:+.2f} m) da {len(best_dE)} sovrapposizioni")
        refined.append(Chunk(
            image=c_b.image,
            coverage=c_b.coverage,
            min_E=c_b.min_E + dE,
            max_N=c_b.max_N + dN,
            gsd_m_per_px=c_b.gsd_m_per_px,
            pass_index=c_b.pass_index,
            frame_indices=c_b.frame_indices,
        ))
    return refined
