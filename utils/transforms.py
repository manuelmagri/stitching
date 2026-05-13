"""Costruzione della similarity 2D che porta una immagine sul canvas a partire da GPS e yaw."""
import math

import numpy as np


def initial_transform(
    record: dict,
    to_utm,
    ref_origin_m: tuple[float, float],
    gsd_canvas: float,
    gsd_frame: float,
    image_size_px: tuple[int, int],
) -> np.ndarray:
    """3x3 similarity: pixel immagine -> pixel canvas.

    - GPS del frame proietta il centro immagine alla coordinata UTM corrispondente.
    - Yaw (corretto per il roll-180 del gimbal) ruota l'immagine.
    - scale = gsd_frame / gsd_canvas compensa l'altitudine variabile.
    - Y del canvas cresce verso sud, quindi north -> -y.
    """
    east, north = to_utm.transform(record["lon"], record["lat"])
    cx_px = (east - ref_origin_m[0]) / gsd_canvas
    cy_px = (ref_origin_m[1] - north) / gsd_canvas

    # FlightYawDegree e' la heading del drone (compass). Lo usiamo direttamente: e' stabile e
    # immune all'ambiguita' del gimbal lock al pitch=-90 (dove gimbal_yaw e gimbal_roll possono
    # rappresentare la stessa orientazione fisica con due set di valori).
    yaw = record["flight_yaw_deg"]
    theta = math.radians(yaw)
    c, s = math.cos(theta), math.sin(theta)
    scale = gsd_frame / gsd_canvas

    w, h = image_size_px
    T_center = np.array(
        [[1.0, 0.0, -w / 2.0], [0.0, 1.0, -h / 2.0], [0.0, 0.0, 1.0]]
    )
    RS = np.array(
        [[scale * c, -scale * s, 0.0], [scale * s, scale * c, 0.0], [0.0, 0.0, 1.0]]
    )
    T_pos = np.array([[1.0, 0.0, cx_px], [0.0, 1.0, cy_px], [0.0, 0.0, 1.0]])
    return T_pos @ RS @ T_center
