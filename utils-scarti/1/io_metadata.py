"""Parsing di `metadati.txt` (GPS+IMU per frame) e dei file IMU derivati."""
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


@dataclass
class FrameMeta:
    source: str            # path JPG originale
    lat: float             # gradi decimali (positivo N)
    lon: float             # gradi decimali (positivo E)
    alt: float             # metri (AMSL)
    flight_yaw: float      # deg, asse Z drone (heading)
    flight_pitch: float
    flight_roll: float
    gimbal_yaw: float
    gimbal_pitch: float
    gimbal_roll: float
    speed: np.ndarray      # vettore velocita' XYZ (m/s) frame drone
    timestamp: datetime


_DMS_RE = re.compile(r"\s*(-?\d+)\s*deg\s*(\d+)'\s*([\d.]+)\"\s*([NSEW])?")


def _parse_dms(s: str) -> float:
    """Converte una stringa GPS DJI tipo `42 deg 59' 36.24" N` in gradi decimali."""
    m = _DMS_RE.match(s)
    if not m:
        raise ValueError(f"Formato GPS non riconosciuto: {s!r}")
    deg, minutes, sec, hemi = m.group(1), m.group(2), m.group(3), m.group(4)
    val = float(deg) + float(minutes) / 60.0 + float(sec) / 3600.0
    if hemi in ("S", "W"):
        val = -val
    return val


def _to_float(x) -> float:
    """Tollera valori EXIF/XMP sia numerici sia stringhe `+43.80` o `81.191 m`."""
    s = str(x).replace("+", "").strip()
    if s.endswith(" m"):
        s = s[:-2].strip()
    return float(s.split()[0])


def load_metadata(metadati_file: Path) -> list[FrameMeta]:
    with open(metadati_file) as f:
        data = json.load(f)
    out: list[FrameMeta] = []
    for e in data:
        lat_s = e.get("XMP:GPSLatitude") or e["EXIF:GPSLatitude"]
        lon_s = e.get("XMP:GPSLongitude") or e["EXIF:GPSLongitude"]
        lat = _parse_dms(lat_s)
        lon = _parse_dms(lon_s)
        alt = _to_float(e["EXIF:GPSAltitude"])
        ts = datetime.strptime(str(e["EXIF:DateTimeOriginal"]), "%Y:%m:%d %H:%M:%S")
        speed = np.array([
            _to_float(e["XMP:FlightXSpeed"]),
            _to_float(e["XMP:FlightYSpeed"]),
            _to_float(e["XMP:FlightZSpeed"]),
        ])
        out.append(FrameMeta(
            source=e["SourceFile"],
            lat=lat, lon=lon, alt=alt,
            flight_yaw=_to_float(e["XMP:FlightYawDegree"]),
            flight_pitch=_to_float(e["XMP:FlightPitchDegree"]),
            flight_roll=_to_float(e["XMP:FlightRollDegree"]),
            gimbal_yaw=_to_float(e["XMP:GimbalYawDegree"]),
            gimbal_pitch=_to_float(e["XMP:GimbalPitchDegree"]),
            gimbal_roll=_to_float(e["XMP:GimbalRollDegree"]),
            speed=speed,
            timestamp=ts,
        ))
    return out


def load_scales(path: Path) -> np.ndarray:
    """Carica `data/scales.txt` (linee tipo `Scale 1: 0.1`) in un array float."""
    out: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Scale"):
                out.append(float(line.split(":")[1].strip()))
    return np.asarray(out, dtype=np.float64)


def load_yaws(path: Path) -> np.ndarray:
    out: list[float] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Yaw"):
                out.append(float(line.split(":")[1].strip()))
    return np.asarray(out, dtype=np.float64)
