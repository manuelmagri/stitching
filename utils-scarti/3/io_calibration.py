"""Caricamento di calibrazione, scale e immagini dal disco."""
from dataclasses import dataclass
import glob
import json
import math
import re

import cv2
import numpy as np

from .timing import timed


@dataclass
class FrameGeometry:
    """Geometria per-frame ricavata dai metadati EXIF/XMP.

    Tutti gli array hanno shape (N,) o (N, 3) e sono allineati per indice.

    Campi:
        positions_enu: (N, 3) coordinate ENU (est, nord, up) in metri, origine
            sul primo frame della selezione.
        alt_asl: (N,) altitudine sopra il livello del mare in metri.
        yaw_deg: (N,) yaw in gradi che ruota l'immagine attorno all'asse verticale,
            in convenzione compass (+cw da nord). Letto da `XMP:FlightYawDegree`
            perché `XMP:GimbalYawDegree` su questo dataset DJI ha un bug di
            logging: salta intermittentemente di 180° tra frame consecutivi.
            Dal momento che il gimbal è solidale al body del drone (FlightYaw
            stabile a 0.3° in crociera), FlightYaw è una proxy più affidabile.
    """
    positions_enu: np.ndarray
    alt_asl: np.ndarray
    yaw_deg: np.ndarray


@timed
def load_calibration(filepath):
    """Legge cameraMatrix e projMatrix da un file JSON (.txt)."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    camera_matrix = np.array(data["cameraMatrix"], dtype=np.float64)
    proj_matrix = np.array(data["projMatrix"], dtype=np.float64)
    return camera_matrix, proj_matrix


@timed
def load_images(pattern, resize_factor=1.0, start=0, end=0):
    """Carica e ridimensiona le immagini in ordine lessicografico.

    start=end=0 → tutte; start>0 ed end=0 → da start in poi; altrimenti [start:end].
    Ritorna (images_bgr, paths).
    """
    paths = sorted(glob.glob(str(pattern)))
    if start == 0 and end == 0:
        selected = paths
    elif start != 0 and end == 0:
        selected = paths[start:]
    else:
        selected = paths[start:end]

    images = []
    for p in selected:
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Impossibile leggere l'immagine: {p}")
        if resize_factor != 1.0:
            h, w = img.shape[:2]
            img = cv2.resize(
                img,
                (int(w / resize_factor), int(h / resize_factor)),
                interpolation=cv2.INTER_AREA,
            )
        images.append(img)
    return images, selected


_DMS_RE = re.compile(
    r"(?P<deg>[-+]?\d+(?:\.\d+)?)\s*deg\s*"
    r"(?P<min>\d+(?:\.\d+)?)'?\s*"
    r"(?P<sec>\d+(?:\.\d+)?)\"?\s*"
    r"(?P<hemi>[NSEW])?",
    re.IGNORECASE,
)


def _dms_to_decimal(value):
    """Converte una stringa DMS tipo "42 deg 59' 36.24\" N" in gradi decimali.

    Accetta anche un float già in gradi decimali (passa-attraverso).
    """
    if isinstance(value, (int, float)):
        return float(value)
    m = _DMS_RE.search(str(value))
    if not m:
        raise ValueError(f"Formato GPS non riconosciuto: {value!r}")
    deg = float(m.group("deg"))
    minutes = float(m.group("min"))
    sec = float(m.group("sec"))
    decimal = abs(deg) + minutes / 60.0 + sec / 3600.0
    if deg < 0:
        decimal = -decimal
    hemi = (m.group("hemi") or "").upper()
    if hemi in ("S", "W"):
        decimal = -decimal
    return decimal


def _altitude_to_meters(value):
    """Estrae il valore numerico (m) da una stringa tipo "81.191 m"."""
    if isinstance(value, (int, float)):
        return float(value)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if not m:
        return 0.0
    return float(m.group(0))


def _slice_metadata(metadata, start, end):
    """Replica la stessa logica di slicing usata da load_images / load_gps_path."""
    if start == 0 and end == 0:
        return metadata
    if start != 0 and end == 0:
        return metadata[start:]
    return metadata[start:end]


def _parse_yaw_deg(value):
    """Estrae lo yaw in gradi accettando float o stringa tipo "+42.80"."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


@timed
def load_gps_path(filepath, start=0, end=0):
    """Carica le coordinate GPS dai metadati e le converte in ENU metrico.

    L'origine ENU è il primo frame nello slice `[start:end]` (stessa logica di
    `load_images`), quindi il path inizia in (0, 0, 0). Asse X = est, Y = nord,
    Z = altitudine relativa. Approssimazione equirettangolare alla latitudine
    di riferimento (accurata su scale << 100 km).

    Ritorna una lista di tuple (east, north, up) in metri.
    """
    with open(filepath, "r") as f:
        metadata = json.load(f)
    selected = _slice_metadata(metadata, start, end)
    if not selected:
        return []

    R_earth = 6378137.0  # WGS84 semi-major axis (m)
    lat0 = _dms_to_decimal(selected[0].get("XMP:GPSLatitude") or selected[0]["EXIF:GPSLatitude"])
    lon0 = _dms_to_decimal(selected[0].get("XMP:GPSLongitude") or selected[0]["EXIF:GPSLongitude"])
    alt0 = _altitude_to_meters(selected[0].get("EXIF:GPSAltitude", 0.0))
    cos_lat0 = math.cos(math.radians(lat0))

    path = []
    for entry in selected:
        lat = _dms_to_decimal(entry.get("XMP:GPSLatitude") or entry["EXIF:GPSLatitude"])
        lon = _dms_to_decimal(entry.get("XMP:GPSLongitude") or entry["EXIF:GPSLongitude"])
        alt = _altitude_to_meters(entry.get("EXIF:GPSAltitude", 0.0))
        east = math.radians(lon - lon0) * cos_lat0 * R_earth
        north = math.radians(lat - lat0) * R_earth
        up = alt - alt0
        path.append((east, north, up))
    return path


@timed
def load_frame_geometry(filepath, start=0, end=0):
    """Estrae (ENU, altitudine ASL, gimbal yaw) per ogni frame nello slice.

    Origine ENU: primo frame nello slice. Stessa convenzione di `load_gps_path`,
    ma in più ritorna lo yaw del gimbal (necessario per l'ortomosaico, perché
    è la rotazione effettiva dell'immagine attorno all'asse verticale).

    Ritorna `FrameGeometry` con array numpy allineati. Verifica che il gimbal
    pitch sia ~ -90° (nadir): se si discosta di più di 5°, stampa un warning
    perché l'assunzione di proiezione ortografica non regge.
    """
    with open(filepath, "r") as f:
        metadata = json.load(f)
    selected = _slice_metadata(metadata, start, end)
    if not selected:
        return FrameGeometry(
            positions_enu=np.zeros((0, 3)),
            alt_asl=np.zeros(0),
            yaw_deg=np.zeros(0),
        )

    R_earth = 6378137.0
    lat0 = _dms_to_decimal(selected[0].get("XMP:GPSLatitude") or selected[0]["EXIF:GPSLatitude"])
    lon0 = _dms_to_decimal(selected[0].get("XMP:GPSLongitude") or selected[0]["EXIF:GPSLongitude"])
    alt0 = _altitude_to_meters(selected[0].get("EXIF:GPSAltitude", 0.0))
    cos_lat0 = math.cos(math.radians(lat0))

    positions = []
    altitudes = []
    yaws = []
    pitches = []
    for entry in selected:
        lat = _dms_to_decimal(entry.get("XMP:GPSLatitude") or entry["EXIF:GPSLatitude"])
        lon = _dms_to_decimal(entry.get("XMP:GPSLongitude") or entry["EXIF:GPSLongitude"])
        alt = _altitude_to_meters(entry.get("EXIF:GPSAltitude", 0.0))
        east = math.radians(lon - lon0) * cos_lat0 * R_earth
        north = math.radians(lat - lat0) * R_earth
        up = alt - alt0
        positions.append((east, north, up))
        altitudes.append(alt)
        # Vedi docstring di FrameGeometry per perché usiamo FlightYaw, non GimbalYaw.
        yaws.append(_parse_yaw_deg(entry["XMP:FlightYawDegree"]))
        pitches.append(_parse_yaw_deg(entry["XMP:GimbalPitchDegree"]))

    pitches_arr = np.asarray(pitches)
    deviation = np.max(np.abs(pitches_arr - (-90.0)))
    if deviation > 5.0:
        print(
            f"[ortho] WARNING: gimbal pitch devia da -90° fino a {deviation:.1f}°; "
            f"l'assunzione nadir non regge, l'ortomosaico avrà distorsioni prospettiche."
        )

    return FrameGeometry(
        positions_enu=np.asarray(positions),
        alt_asl=np.asarray(altitudes),
        yaw_deg=np.asarray(yaws),
    )
