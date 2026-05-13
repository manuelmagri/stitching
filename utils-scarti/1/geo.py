"""Conversioni geografiche e allineamento similarity (Umeyama)."""
from __future__ import annotations

import numpy as np

try:
    from pyproj import Transformer
except ImportError as _e:  # delayed: la pipeline funziona anche senza UTM esplicito
    Transformer = None  # type: ignore


def utm_zone_for(lon_deg: float) -> int:
    return int((lon_deg + 180.0) / 6.0) + 1


def utm_epsg(lat_deg: float, lon_deg: float) -> int:
    """Restituisce l'EPSG UTM (WGS84) corretto per la latitudine/longitudine date."""
    zone = utm_zone_for(lon_deg)
    return (32600 if lat_deg >= 0 else 32700) + zone


def latlon_to_utm(lats: np.ndarray, lons: np.ndarray) -> tuple[np.ndarray, int]:
    """Converte arrays di lat/lon (deg) in coordinate UTM (E, N) in metri.

    Ritorna `((N, 2) array, epsg_code)`. Tutti i punti vengono proiettati nello
    stesso fuso UTM (quello del primo punto), che e' adeguato per voli di
    qualche centinaio di metri.
    """
    if Transformer is None:
        raise RuntimeError("pyproj non disponibile: aggiungilo a requirements.txt")
    lats = np.asarray(lats, dtype=np.float64).ravel()
    lons = np.asarray(lons, dtype=np.float64).ravel()
    epsg = utm_epsg(float(lats[0]), float(lons[0]))
    t = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    e, n = t.transform(lons, lats)
    return np.column_stack([np.asarray(e, dtype=np.float64), np.asarray(n, dtype=np.float64)]), epsg


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Stima la similarity transform ottimale (least-squares) src -> dst.

    Riferimento: Umeyama (IEEE TPAMI 1991). Lavora in dimensione arbitraria.

    Ritorna (scala s, matrice di rotazione R, traslazione t) tali che
        dst ~ s * R @ src + t
    Per ogni colonna `src.T[:, i]`.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape, "src e dst devono avere stessa shape"
    n, d = src.shape
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(d)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1.0
    R = U @ D @ Vt
    if with_scale:
        var_s = (src_c ** 2).sum() / n
        s = float((S * np.diag(D)).sum() / var_s) if var_s > 0 else 1.0
    else:
        s = 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_similarity(s: float, R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applica una similarity (s, R, t) a un array (N, d): out = s * R @ pts + t."""
    pts = np.asarray(pts, dtype=np.float64)
    return (s * (R @ pts.T)).T + t
