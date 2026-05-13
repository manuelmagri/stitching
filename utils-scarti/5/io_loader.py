"""Caricamento di calibrazione, metadati EXIF/IMU e immagini in streaming.

Punto di ingresso lato pipeline (post-preprocessing). I file letti sono quelli
prodotti da `preprocessing.ensure_preprocessing`:

  - data/calibration.txt              JSON con `cameraMatrix` (3x3) post-undistort
  - data/metadati.txt                 JSON array, un dict per immagine sorgente
  - immagini/.../immagini_senza_distorsione/  .jpg undistorted

I metadati sono globali (tutto il volo), le immagini undistorted possono essere
solo un sotto-intervallo: l'accoppiamento avviene per basename del file.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Tipi pubblici
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CameraCalibration:
    """Intrinseci della camera *post-undistort*, gia' allineati ai file .jpg."""

    K: np.ndarray                              # 3x3 float64
    proj_matrix: np.ndarray | None = None      # 3x4 K @ extr (gimbal), opzionale


@dataclass(frozen=True)
class FrameMeta:
    """Metadati di un singolo scatto, sufficienti per VO + fusione + mosaic.

    Le grandezze IMU sono in gradi; le velocita' in m/s; il timestamp e' un
    datetime con risoluzione al secondo (EXIF non offre piu' fine).
    """

    index: int                       # indice globale nell'elenco sorgente ordinato
    image_path: str                  # path assoluto della .jpg undistorted
    timestamp: datetime

    lat_deg: float
    lon_deg: float
    alt_m: float

    flight_yaw_deg: float
    flight_pitch_deg: float
    flight_roll_deg: float

    gimbal_yaw_deg: float
    gimbal_pitch_deg: float
    gimbal_roll_deg: float

    flight_speed_ms: np.ndarray = field(default_factory=lambda: np.zeros(3))  # (vx, vy, vz)


# --------------------------------------------------------------------------- #
# Calibrazione
# --------------------------------------------------------------------------- #

def carica_calibrazione(calib_path: str) -> CameraCalibration:
    """Legge `calibration.txt` e ritorna gli intrinseci come ndarray.

    Il file e' scritto da `preprocessing.ensure_preprocessing` con la K
    *effettiva* dopo undistort + crop alla ROI. Tutto il resto della pipeline
    deve usare ESCLUSIVAMENTE questa K (la K sorgente XMP non corrisponde piu'
    alla geometria dei pixel salvati).
    """
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"Calibrazione non trovata: {calib_path}. "
            "Esegui `preprocessing.ensure_preprocessing` prima della pipeline."
        )
    with open(calib_path, "r") as f:
        blob = json.load(f)

    if "cameraMatrix" not in blob:
        raise ValueError(f"Campo `cameraMatrix` mancante in {calib_path}")
    K = np.asarray(blob["cameraMatrix"], dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"`cameraMatrix` non 3x3 ma {K.shape} in {calib_path}")

    proj = blob.get("projMatrix")
    proj_arr = None
    if proj is not None:
        proj_arr = np.asarray(proj, dtype=np.float64)
        if proj_arr.shape != (3, 4):
            # Non blocchiamo la pipeline: la proj e' usata solo dalla
            # decomposizione essenziale "vecchia" e puo' essere ricostruita.
            proj_arr = None

    return CameraCalibration(K=K, proj_matrix=proj_arr)


# --------------------------------------------------------------------------- #
# Metadati EXIF/IMU
# --------------------------------------------------------------------------- #

# exiftool, invocato senza `-n`, restituisce GPSLatitude in EXIF come stringa
# DMS ("45 deg 5' 23.45\" N"), in XMP come float firmato. Supportiamo entrambi.
def _parse_gps_value(raw, hemisphere_axis: str) -> float:
    """Converte un valore GPS exiftool (XMP float o EXIF DMS) in gradi decimali.

    `hemisphere_axis` e' "lat" o "lon": serve solo se `raw` e' un float gia'
    senza segno (alcuni firmware DJI usano XMP gia' firmato, altri no).
    """
    if isinstance(raw, (int, float)):
        return float(raw)

    s = str(raw).strip()
    if not s:
        raise ValueError("Stringa GPS vuota")

    # Forma DMS: "45 deg 5' 23.45\" N" oppure "12 deg 30' 5\" E"
    sign = 1.0
    if s.endswith(("S", "W")):
        sign = -1.0
        s = s[:-1].strip()
    elif s.endswith(("N", "E")):
        s = s[:-1].strip()

    # Tokeni semplici: rimuove unita' simboliche, lascia i numeri.
    for tok in ("deg", "°", "'", "\""):
        s = s.replace(tok, " ")
    parts = [p for p in s.split() if p]

    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"GPS non parseable: {raw!r}") from exc

    if len(nums) == 1:
        # Gia' decimale, ma potrebbe servire il segno.
        val = nums[0]
    elif len(nums) == 2:
        val = nums[0] + nums[1] / 60.0
    elif len(nums) == 3:
        val = nums[0] + nums[1] / 60.0 + nums[2] / 3600.0
    else:
        raise ValueError(f"GPS con {len(nums)} componenti: {raw!r}")

    return sign * val


def _parse_altitude(raw) -> float:
    """GPSAltitude puo' essere float oppure stringa tipo '123.4 m'."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip().lower().replace("m", "").strip()
    # Alcuni file portano l'indicazione 'Above Sea Level'/'Below Sea Level'.
    sign = -1.0 if "below" in s else 1.0
    s = s.replace("above sea level", "").replace("below sea level", "").strip()
    return sign * float(s.split()[0])


def _parse_timestamp(raw) -> datetime:
    """EXIF DateTimeOriginal: 'YYYY:MM:DD HH:MM:SS' (separatore `:` anche in data)."""
    return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")


def _get_any(entry: dict, *candidate_keys: str):
    """Recupera il primo campo presente tra `candidate_keys`. None se nessuno."""
    for k in candidate_keys:
        if k in entry:
            return entry[k]
    return None


def _frame_meta_from_entry(entry: dict, index: int, image_path: str) -> FrameMeta:
    """Costruisce un FrameMeta da un dict exiftool, dopo aver risolto il path."""

    # GPS: XMP DJI e' piu' affidabile (gia' decimale e firmato); EXIF e' fallback.
    lat_raw = _get_any(entry, "XMP:GPSLatitude", "EXIF:GPSLatitude",
                       "Composite:GPSLatitude")
    lon_raw = _get_any(entry, "XMP:GPSLongitude", "EXIF:GPSLongitude",
                       "Composite:GPSLongitude")
    alt_raw = _get_any(entry, "XMP:GPSAltitude", "EXIF:GPSAltitude",
                       "Composite:GPSAltitude", "XMP:AbsoluteAltitude",
                       "XMP:RelativeAltitude")
    if lat_raw is None or lon_raw is None or alt_raw is None:
        raise ValueError(
            f"GPS mancante per frame {index} ({os.path.basename(image_path)}): "
            f"lat={lat_raw}, lon={lon_raw}, alt={alt_raw}"
        )

    lat = _parse_gps_value(lat_raw, "lat")
    lon = _parse_gps_value(lon_raw, "lon")
    alt = _parse_altitude(alt_raw)

    ts = _parse_timestamp(_get_any(entry, "EXIF:DateTimeOriginal",
                                   "XMP:DateTimeOriginal"))

    def _f(key: str, default: float = 0.0) -> float:
        v = entry.get(key)
        return float(v) if v is not None else default

    speed = np.array([
        _f("XMP:FlightXSpeed"),
        _f("XMP:FlightYSpeed"),
        _f("XMP:FlightZSpeed"),
    ], dtype=np.float64)

    return FrameMeta(
        index=index,
        image_path=image_path,
        timestamp=ts,
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt,
        flight_yaw_deg=_f("XMP:FlightYawDegree"),
        flight_pitch_deg=_f("XMP:FlightPitchDegree"),
        flight_roll_deg=_f("XMP:FlightRollDegree"),
        gimbal_yaw_deg=_f("XMP:GimbalYawDegree"),
        gimbal_pitch_deg=_f("XMP:GimbalPitchDegree"),
        gimbal_roll_deg=_f("XMP:GimbalRollDegree"),
        flight_speed_ms=speed,
    )


def _basename_index(entries: list[dict]) -> dict[str, tuple[int, dict]]:
    """Mappa basename .jpg -> (indice_globale, entry).

    L'indice globale e' la posizione nell'elenco sorgente ordinato per nome,
    coerente con quello usato da `preprocessing.ensure_preprocessing` quando
    seleziona il sotto-intervallo da undistortare.
    """
    # Ordino qui per garantire l'indice anche se `metadati.txt` non e' ordinato.
    ordered = sorted(entries, key=lambda e: os.path.basename(e.get("SourceFile", "")))
    out: dict[str, tuple[int, dict]] = {}
    for i, e in enumerate(ordered):
        src = e.get("SourceFile")
        if not src:
            continue
        out[os.path.basename(src)] = (i, e)
    return out


def carica_frames_meta(metadata_path: str,
                       undistort_dir: str,
                       start: int = 0,
                       end: int | None = None) -> list[FrameMeta]:
    """Carica i metadati e li allinea ai .jpg presenti in `undistort_dir`.

    Solo i frame il cui basename esiste in `undistort_dir` entrano nel
    risultato. L'`index` di FrameMeta resta quello GLOBALE (cioe' la
    posizione nell'elenco sorgente), cosi' i log restano coerenti anche
    quando si lavora su un sotto-intervallo.

    `start` e `end` sono interpretati nello spazio degli indici GLOBALI
    e restringono ulteriormente la selezione.
    """
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f"Metadati non trovati: {metadata_path}. "
            "Esegui `preprocessing.ensure_preprocessing` prima della pipeline."
        )
    if not os.path.isdir(undistort_dir):
        raise FileNotFoundError(f"Cartella undistorted non trovata: {undistort_dir}")

    with open(metadata_path, "r") as f:
        entries = json.load(f)
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"`{metadata_path}` non e' un array JSON non vuoto")

    by_basename = _basename_index(entries)

    # File undistorted disponibili, in ordine alfabetico (== ordine globale).
    jpg_files = sorted(
        glob.glob(os.path.join(undistort_dir, "*.[jJ][pP][gG]"))
    )

    frames: list[FrameMeta] = []
    for path in jpg_files:
        bn = os.path.basename(path)
        if bn not in by_basename:
            # Undistorted senza metadati corrispondenti: skip silenzioso.
            continue
        idx, entry = by_basename[bn]
        if idx < start:
            continue
        if end is not None and idx >= end:
            continue
        frames.append(_frame_meta_from_entry(entry, idx, path))

    if not frames:
        raise RuntimeError(
            f"Nessun frame valido in [{start}:{end}] dopo l'accoppiamento "
            f"con {undistort_dir}. Hai eseguito undistort sull'intervallo giusto?"
        )

    # L'ordine globale e' gia' garantito da `jpg_files`; lo ribadiamo per
    # robustezza in caso di basename non monotoni.
    frames.sort(key=lambda fm: fm.index)
    return frames


# --------------------------------------------------------------------------- #
# Stream immagini lazy
# --------------------------------------------------------------------------- #

class FrameStream:
    """Iteratore lazy sulle immagini: `cv2.imread` un frame alla volta.

    Pensato per cicli VO/mosaic che mantengono in memoria al piu' il frame
    corrente e quello precedente. Per accessi puntuali (es. ricaricare il
    frame i durante un fallback) usare `read(i)`.
    """

    def __init__(self, frames: list[FrameMeta]):
        if not frames:
            raise ValueError("FrameStream creato con lista vuota")
        self._frames = frames

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def metas(self) -> list[FrameMeta]:
        """Lista (read-only logicamente) dei FrameMeta sottostanti."""
        return self._frames

    def read(self, i: int) -> tuple[FrameMeta, np.ndarray]:
        """Legge il frame in posizione `i` (0-based nel sotto-intervallo)."""
        fm = self._frames[i]
        img = cv2.imread(fm.image_path)
        if img is None:
            raise IOError(f"Lettura fallita: {fm.image_path}")
        return fm, img

    def __iter__(self) -> Iterator[tuple[FrameMeta, np.ndarray]]:
        for i in range(len(self._frames)):
            yield self.read(i)
