"""Caricamento degli input: calibrazione, metadati di volo, percorsi delle immagini.

La calibrazione descrive le immagini RETTIFICATE prodotte da `preprocessing.undistort_images`,
non gli scatti originali: focale, centro ottico e dimensione cambiano col ritaglio.
`verify_image_size` confronta la dimensione dichiarata con i file su disco e fallisce
subito se non corrispondono, perche' la focale e' l'unico numero da cui dipende la scala
metrica e uno scarto li' sfasa in silenzio footprint, overlap e mosaico.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

_FRAME_RE = re.compile(r"_(\d{4})_D\.JPG", re.IGNORECASE)


@dataclass(frozen=True)
class Calibration:
    """Intrinseci delle immagini rettificate, piu' la dimensione che devono avere."""

    camera_matrix: np.ndarray
    image_size: tuple[int, int]  # (larghezza, altezza)

    @property
    def focal_px(self) -> float:
        return float(self.camera_matrix[0, 0])


def load_calibration(calibration_path: Path) -> Calibration:
    with open(calibration_path) as f:
        data = json.load(f)

    if "imageSize" not in data:
        raise ValueError(
            f"{calibration_path} non contiene 'imageSize'. E' una calibrazione scritta da "
            "una versione precedente di preprocessing.create_calibration, che salvava gli "
            "intrinseci ORIGINALI invece di quelli rettificati: rigenerala."
        )

    w, h = data["imageSize"]
    return Calibration(
        camera_matrix=np.array(data["cameraMatrix"], dtype=np.float64),
        image_size=(int(w), int(h)),
    )


def verify_image_size(calibration: Calibration, sample_path: Path) -> None:
    """Alza ValueError se `sample_path` non ha la dimensione dichiarata dalla calibrazione."""
    img = cv2.imread(str(sample_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Impossibile leggere l'immagine di controllo {sample_path}")

    trovata = (img.shape[1], img.shape[0])
    if trovata != calibration.image_size:
        raise ValueError(
            f"La calibrazione dichiara immagini {calibration.image_size[0]}x"
            f"{calibration.image_size[1]}, ma {sample_path.name} e' {trovata[0]}x{trovata[1]}. "
            "Calibrazione e immagini rettificate non provengono dalla stessa esecuzione: "
            "rigenera la calibrazione con preprocessing.create_calibration."
        )


def _parse_dms(value: str) -> float:
    s = value.replace('"', "").replace("'", "").replace("deg", "")
    parts = s.split()
    return float(parts[0]) + float(parts[1]) / 60.0 + float(parts[2]) / 3600.0


def _parse_signed_number(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).replace("+", "").strip())


def load_records(metadata_path: Path) -> list[dict]:
    """Un record per scatto, ordinati per indice di frame.

    Il campo `vo_idx` e' la posizione del record in questa lista completa: e' l'indice con
    cui indirizzare data/translations.json, che ha una riga per scatto nello stesso ordine.
    Va assegnato qui, PRIMA di qualunque filtro sul range, altrimenti selezionare un
    sottoinsieme di frame disallineerebbe l'odometria.
    """
    with open(metadata_path) as f:
        raw = json.load(f)

    records: list[dict] = []
    for entry in raw:
        match = _FRAME_RE.search(entry.get("SourceFile", ""))
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
                "relative_alt_m": _parse_signed_number(entry["XMP:RelativeAltitude"]),
                "gimbal_pitch_deg": _parse_signed_number(entry["XMP:GimbalPitchDegree"]),
                "flight_yaw_deg": _parse_signed_number(entry["XMP:FlightYawDegree"]),
            }
        )

    records.sort(key=lambda r: r["index"])
    for vo_idx, record in enumerate(records):
        record["vo_idx"] = vo_idx
    return records


def list_image_paths(images_dir: Path) -> dict[int, Path]:
    """Mappa indice di frame -> percorso, per le .jpg con nome ..._NNNN_D.JPG."""
    out: dict[int, Path] = {}
    for path in images_dir.iterdir():
        if path.suffix.lower() != ".jpg":
            continue
        match = _FRAME_RE.search(path.name)
        if match:
            out[int(match.group(1))] = path
    return out


def select_range(
    records: list[dict],
    image_paths: dict[int, Path],
    frame_start: int | None = None,
    frame_end: int | None = None,
) -> list[dict]:
    """Record che hanno sia metadati sia file immagine, filtrati sul range richiesto.

    Aggiunge `path` a ogni record tenuto. `vo_idx` resta quello della lista completa.
    """
    if not image_paths:
        raise ValueError("Nessuna immagine trovata nella cartella del volo")

    start = frame_start if frame_start is not None else min(image_paths)
    end = frame_end if frame_end is not None else max(image_paths)

    kept: list[dict] = []
    for record in records:
        if not start <= record["index"] <= end:
            continue
        path = image_paths.get(record["index"])
        if path is None:
            continue
        record["path"] = path
        kept.append(record)

    if not kept:
        raise ValueError(f"Nessun frame con metadati e immagine nel range {start}..{end}")
    return kept


def altitude_m(record: dict) -> float:
    """Quota da usare per il GSD: la barometrica relativa, non la GPS.

    Su un volo di prova RelativeAltitude ha deviazione standard 0,05 m su 835 scatti,
    contro 0,26 m di GPSAltitude.
    """
    return abs(float(record["relative_alt_m"]))
