"""Caricamento di calibrazione, metadati GPS e immagini, con filtro per range di frame."""
import json
import re
from pathlib import Path

import numpy as np

_FRAME_RE = re.compile(r"_(\d{4})_D\.JPG", re.IGNORECASE)


def load_camera_matrix(calibration_path: Path) -> np.ndarray:
    with open(calibration_path) as f:
        data = json.load(f)
    return np.array(data["cameraMatrix"], dtype=np.float64)


def _parse_dms(value: str) -> float:
    s = value.replace('"', "").replace("'", "").replace("deg", "")
    parts = s.split()
    return float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0


def _parse_signed_number(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("+", "").strip())


def _parse_altitude(value) -> float:
    s = str(value).split()[0]
    return float(s)


def load_metadata(metadata_path: Path) -> list[dict]:
    """Restituisce una lista di record ordinati per frame index, ognuno con GPS, yaw, roll e quota."""
    with open(metadata_path) as f:
        raw = json.load(f)

    records: list[dict] = []
    for entry in raw:
        src = entry.get("SourceFile", "")
        match = _FRAME_RE.search(src)
        if not match:
            continue

        lat = _parse_dms(entry["EXIF:GPSLatitude"])
        if "S" in str(entry.get("XMP:GPSLatitude", "")):
            lat = -lat
        lon = _parse_dms(entry["EXIF:GPSLongitude"])
        if "W" in str(entry.get("XMP:GPSLongitude", "")):
            lon = -lon

        records.append(
            {
                "index": int(match.group(1)),
                "lat": lat,
                "lon": lon,
                "altitude_m": _parse_altitude(entry["EXIF:GPSAltitude"]),
                "relative_alt_m": _parse_signed_number(entry["XMP:RelativeAltitude"]),
                "gimbal_yaw_deg": _parse_signed_number(entry["XMP:GimbalYawDegree"]),
                "gimbal_pitch_deg": _parse_signed_number(entry["XMP:GimbalPitchDegree"]),
                "gimbal_roll_deg": _parse_signed_number(entry["XMP:GimbalRollDegree"]),
                "flight_yaw_deg": _parse_signed_number(entry["XMP:FlightYawDegree"]),
                "flight_pitch_deg": _parse_signed_number(entry["XMP:FlightPitchDegree"]),
                "flight_roll_deg": _parse_signed_number(entry["XMP:FlightRollDegree"]),
            }
        )

    records.sort(key=lambda r: r["index"])
    return records


def list_image_paths(images_dir: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in images_dir.iterdir():
        if p.suffix.lower() != ".jpg":
            continue
        m = _FRAME_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def select_range(
    records: list[dict],
    image_paths: list[tuple[int, Path]],
    frame_start: int,
    frame_end: int | None,
) -> list[dict]:
    """Tiene solo i frame che hanno sia metadati che file immagine ed esegue il filtro sul range."""
    if frame_end is None:
        frame_end = max(idx for idx, _ in image_paths)

    image_map = {idx: p for idx, p in image_paths if frame_start <= idx <= frame_end}
    keep: list[dict] = []
    for r in records:
        if r["index"] < frame_start or r["index"] > frame_end:
            continue
        path = image_map.get(r["index"])
        if path is None:
            continue
        r["path"] = path
        keep.append(r)
    return keep
