"""Caricamento di calibrazione, metadati EXIF/IMU e immagini undistorted.

Le immagini vengono esposte come iteratore lazy (una sola in RAM alla volta)
per evitare il pattern di OLD-CODE che caricava tutto in memoria all'avvio.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Calibrazione
# ---------------------------------------------------------------------------

def carica_calibrazione(data_dir: str):
    """Legge cameraMatrix e projMatrix da `data_dir/calibration.txt`.

    Ritorna (camera_matrix 3x3, proj_matrix 3x4) come np.ndarray float64.
    """
    path = os.path.join(data_dir, "calibration.txt")
    with open(path, "r") as f:
        blob = json.load(f)
    K = np.asarray(blob["cameraMatrix"], dtype=np.float64)
    P = np.asarray(blob["projMatrix"], dtype=np.float64)
    return K, P


# ---------------------------------------------------------------------------
# Metadati EXIF/IMU
# ---------------------------------------------------------------------------

@dataclass
class FrameMeta:
    """Metadati di un singolo scatto, normalizzati e tipizzati."""

    source_file: str
    lat_deg: float           # latitudine WGS84 con segno (N+, S-)
    lon_deg: float           # longitudine WGS84 con segno (E+, W-)
    alt_m: float             # altitudine ortometrica (m s.l.m.)
    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float
    flight_yaw_deg: float
    flight_pitch_deg: float
    flight_roll_deg: float
    flight_speed: np.ndarray  # 3-vector m/s in body frame DJI
    timestamp: datetime


_DMS_RE = re.compile(
    r"\s*([+-]?\d+(?:\.\d+)?)\s*deg\s*([+-]?\d+(?:\.\d+)?)\s*'\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*\"?\s*([NSEW])?\s*"
)


def _to_float(value) -> float:
    """Converte un valore EXIF/XMP in float, accettando '+12.3', '12.3', 12.3."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip().lstrip("+"))


def _parse_dms(value, hemisphere_default: str | None = None) -> float:
    """Parsa una stringa DMS DJI (es. "42 deg 59' 36.24\" N") in gradi decimali.

    Se `value` e' gia' un numero, lo ritorna. L'emisfero e' opzionale: se
    presente nella stringa lo uso, altrimenti applico `hemisphere_default`.
    Le coordinate sud/ovest sono restituite con segno negativo.
    """
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    m = _DMS_RE.match(text)
    if not m:
        # Eventuali stringhe tipo "+42.99340" senza DMS
        return _to_float(text)

    deg, minutes, seconds, hemi = m.groups()
    decimal = abs(float(deg)) + float(minutes) / 60.0 + float(seconds) / 3600.0
    sign = -1.0 if float(deg) < 0 else 1.0

    hemi = hemi or hemisphere_default
    if hemi in ("S", "W"):
        sign = -1.0
    elif hemi in ("N", "E"):
        sign = 1.0

    return sign * decimal


def _parse_alt(value) -> float:
    """Parsa l'altitudine EXIF: stringhe tipo '81.191 m', numeri puri."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower().replace("m", "").strip()
    return float(text)


def carica_metadati(metadata_file: str) -> list[FrameMeta]:
    """Legge `data/metadati.txt` (JSON da exiftool) e ritorna una lista di FrameMeta.

    L'ordine e' lo stesso del file (gia' ordinato lessicograficamente dal
    preprocessing, equivalente a ordine temporale per i nomi DJI).
    """
    with open(metadata_file, "r") as f:
        raw = json.load(f)

    frames: list[FrameMeta] = []
    for entry in raw:
        # Preferisco i campi XMP che gia' contengono l'emisfero; fallback su EXIF.
        lat_raw = entry.get("XMP:GPSLatitude") or entry.get("EXIF:GPSLatitude")
        lon_raw = entry.get("XMP:GPSLongitude") or entry.get("EXIF:GPSLongitude")
        alt_raw = entry.get("EXIF:GPSAltitude") or entry.get("Composite:GPSAltitude")

        frames.append(FrameMeta(
            source_file=entry["SourceFile"],
            lat_deg=_parse_dms(lat_raw, hemisphere_default="N"),
            lon_deg=_parse_dms(lon_raw, hemisphere_default="E"),
            alt_m=_parse_alt(alt_raw),
            gimbal_yaw_deg=_to_float(entry["XMP:GimbalYawDegree"]),
            gimbal_pitch_deg=_to_float(entry["XMP:GimbalPitchDegree"]),
            gimbal_roll_deg=_to_float(entry["XMP:GimbalRollDegree"]),
            flight_yaw_deg=_to_float(entry["XMP:FlightYawDegree"]),
            flight_pitch_deg=_to_float(entry["XMP:FlightPitchDegree"]),
            flight_roll_deg=_to_float(entry["XMP:FlightRollDegree"]),
            flight_speed=np.array([
                _to_float(entry["XMP:FlightXSpeed"]),
                _to_float(entry["XMP:FlightYSpeed"]),
                _to_float(entry["XMP:FlightZSpeed"]),
            ], dtype=np.float64),
            timestamp=datetime.strptime(
                str(entry["EXIF:DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S"
            ),
        ))

    # EXIF DateTimeOriginal e' arrotondato al secondo: con cadenza di scatto
    # ~1 s ma non perfettamente sincrona, due scatti possono cadere nello stesso
    # secondo (dt=0, due timestamp identici -> a valle collassano nello smoothing)
    # oppure su secondi separati di 2 (dt=2, falsi outlier di velocita').
    # Ricostruisco una scala temporale uniforme t_i = t_0 + i * dt_nominal:
    #   - dt_nominal = mediana dei dt > 0 sulla sequenza EXIF (robusta a 0 e a 2).
    # Cosi' i frame consecutivi hanno sempre dt costante e i moduli a valle
    # (pose_smooth, pose_fusion) non vedono duplicati ne' falsi salti.
    if len(frames) >= 2:
        dts_pos = [
            (frames[i].timestamp - frames[i - 1].timestamp).total_seconds()
            for i in range(1, len(frames))
        ]
        dts_pos = [d for d in dts_pos if d > 0]
        if dts_pos:
            dt_nominal_s = float(np.median(dts_pos))
            t0 = frames[0].timestamp
            for i, fr in enumerate(frames):
                fr.timestamp = t0 + timedelta(seconds=i * dt_nominal_s)

    return frames


# ---------------------------------------------------------------------------
# Immagini undistorted (streaming)
# ---------------------------------------------------------------------------

def _img_index(path: str) -> int:
    """Estrae il numero dal nome DJI (es. ..._0042_D.JPG -> 42).

    Fallback al sort lessicografico tramite -1 se non trova un pattern noto.
    """
    name = os.path.basename(path)
    m = re.search(r"_(\d{4,})_", name)
    if m:
        return int(m.group(1))
    m = re.search(r"img_(\d+)", name)
    return int(m.group(1)) if m else -1


def lista_immagini(undistort_dir: str) -> list[str]:
    """Lista ordinata dei path delle immagini undistorted in `undistort_dir`."""
    paths = glob.glob(os.path.join(undistort_dir, "*.[jJ][pP][gG]"))
    if not paths:
        raise FileNotFoundError(f"Nessuna .jpg in {undistort_dir}")
    paths.sort(key=lambda p: (_img_index(p), p))
    return paths


def carica_immagine(path: str, scale: float = 1.0):
    """Carica una singola immagine in BGR, opzionalmente ridimensionata.

    `scale=1.0` mantiene la risoluzione nativa. `scale=0.5` dimezza i lati.
    Diversamente da OLD-CODE non c'e' un fattore di resize hardcoded: la
    decisione e' del chiamante e visibile a colpo d'occhio.
    """
    img = cv2.imread(path)
    if img is None:
        raise IOError(f"Impossibile leggere {path}")
    if scale != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return img


def itera_immagini(undistort_dir: str, scale: float = 1.0):
    """Generator: produce (index, path, image) per ogni immagine.

    Tenere in memoria una sola immagine alla volta. L'index parte da 0 ed e'
    progressivo nell'ordine dei file (NON il numero DJI nel nome).
    """
    for i, path in enumerate(lista_immagini(undistort_dir)):
        yield i, path, carica_immagine(path, scale=scale)
