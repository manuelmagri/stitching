"""Omografia analitica immagine -> piano-mondo metrico.

Modello "ortofoto piatta", appropriato per droni DJI con gimbal stabilizzato
a nadir e sequenze a quota costante:

    - GSD (m/pixel) = (alt_drone - quota_terreno) / focale_pixel
    - rotazione planare: yaw del drone (DJI: 0 = Nord, +90 = Est)
    - traslazione: il centro dell'immagine cade sulla posizione (x, y)
      della camera nel mondo

L'omografia si riduce quindi a una similitudine 2D in coordinate omogenee.
La componente di non-nadiralita' (pitch/roll del drone, mentre il gimbal e'
stabilizzato) introduce errori di alcuni decimetri al bordo del frame, che
trascuriamo nella prima versione.

Convenzione del frame mondo: X = Est, Y = Nord (UTM-locale), in metri.
Convenzione DJI yaw: 0 deg = Nord, +90 deg = Est, +180 deg = Sud (orario).

Convenzione foto DJI con gimbal nadir e camera "raddrizzata" dal firmware:
    riga in alto  (v=0)         -> direzione di volo del drone
    riga in basso (v=h_max-1)   -> direzione opposta
    colonna a sinistra (u=0)    -> sinistra del drone
    colonna a destra  (u=w_max) -> destra del drone
"""

from __future__ import annotations

import numpy as np

from .io_loader import FrameMeta


def omografia_immagine_su_piano(camera_matrix: np.ndarray,
                                meta: FrameMeta,
                                cam_xy_world: tuple[float, float],
                                ground_z_world: float = 0.0,
                                yaw_deg: float | None = None,
                                agl_m: float | None = None) -> np.ndarray:
    """Omografia che porta i pixel di un'immagine in coordinate metriche del mondo.

    Parametri
    ----------
    camera_matrix : K 3x3 (intrinseci) coerente con la risoluzione dell'immagine
        passata al warp (occhio se l'immagine e' stata ridimensionata: K va scalata).
    meta : FrameMeta del frame; servono `alt_m` e (se `yaw_deg` non e' passato)
        `flight_yaw_deg`.
    cam_xy_world : (x, y) della camera nel mondo (UTM-locale, m).
    ground_z_world : quota del piano del terreno (m, default 0).
    yaw_deg : se fornito, sostituisce `flight_yaw_deg` come orientamento del
        frame. `flight_yaw_deg` e' la scelta di default perche' nel dataset
        in esame il `gimbal_yaw_deg` del primissimo scatto e' inattendibile
        (drone appena decollato).
    agl_m : altezza del drone *sopra il terreno*, in metri. Se fornito
        sostituisce `meta.alt_m - ground_z_world`. Tipicamente arriva
        dall'auto-calibrazione: alt_EXIF e' ASL e puo' essere ben diversa
        dall'AGL effettiva se il terreno non e' a quota 0 m s.l.m.

    Ritorna H 3x3 tale che:
        [Xw, Yw, 1]^T  =  H @ [u, v, 1]^T
    con (u, v) coordinate pixel e (Xw, Yw) coordinate metriche del mondo.
    """
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    if agl_m is not None:
        h = max(float(agl_m), 1e-3)
    else:
        h = max(meta.alt_m - ground_z_world, 1e-3)
    gsd_x = h / fx        # m/pixel lungo l'asse u dell'immagine
    gsd_y = h / fy        # m/pixel lungo l'asse v dell'immagine

    yaw = float(yaw_deg if yaw_deg is not None else meta.flight_yaw_deg)
    yaw_rad = np.deg2rad(yaw)
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)

    # Mappa lo spostamento pixel rispetto al centro (du, dv) in spostamento mondo:
    #   du>0  (a destra del drone)            -> mondo: ( gsd*c, -gsd*s)
    #   dv<0  (in alto = davanti al drone)    -> mondo: ( gsd*s,  gsd*c)
    # In matriciale agendo su [u, v, 1]:
    a_uu =  gsd_x * c
    a_uv = -gsd_y * s         # nota: dv positivo == "indietro", quindi cambia segno
    a_vu = -gsd_x * s
    a_vv = -gsd_y * c

    # Traslazione: il centro ottico (cx, cy) deve mappare su cam_xy_world.
    cam_x, cam_y = float(cam_xy_world[0]), float(cam_xy_world[1])
    tx = cam_x - a_uu * cx - a_uv * cy
    ty = cam_y - a_vu * cx - a_vv * cy

    return np.array([
        [a_uu, a_uv, tx],
        [a_vu, a_vv, ty],
        [0.0,  0.0,  1.0],
    ], dtype=np.float64)
