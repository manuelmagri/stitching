"""Orchestratore minimale: GPS+yaw posiziona ogni frame, niente refinement."""
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from . import flight_filter, georef, io_data, mosaic, transforms
from .geodesy import gsd_meters_per_pixel, make_transformers


def _load_image(path: Path, downscale: float) -> np.ndarray | None:
    """Carica un'immagine e la ridimensiona di downscale. Niente flip: con FlightYaw l'immagine
    e' gia' correttamente orientata (le configurazioni roll=0 e roll=180 sono equivalenti
    geometricamente quando il pitch=-90, e producono lo stesso campionamento pixel).
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if downscale != 1.0:
        nw = int(round(img.shape[1] / downscale))
        nh = int(round(img.shape[0] / downscale))
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return img


def _frame_altitude(rec: dict) -> float:
    rel = rec.get("relative_alt_m")
    return abs(rel) if rel is not None else abs(rec["altitude_m"])


def run_pipeline(cfg: dict) -> None:
    images_dir: Path = cfg["images_dir"]
    output_jpg: Path = cfg["output_jpg"]
    output_tiff: Path = cfg["output_tiff"]
    frame_start: int = cfg["frame_start"]
    frame_end = cfg.get("frame_end")
    downscale: float = cfg.get("downscale", 2.5)

    # 1) Carica tutto il dataset; applica il filtro curve PRIMA del range cosi' i frame in
    #    curva vengono identificati in modo consistente indipendentemente dal range richiesto.
    K = io_data.load_camera_matrix(cfg["calibration_file"])
    records_all = io_data.load_metadata(cfg["metadata_file"])

    if cfg.get("skip_curves", True):
        records_all, stats = flight_filter.filter_curves(
            records_all,
            roll_threshold_deg=cfg.get("roll_threshold_deg", 8.0),
            yaw_rate_threshold_deg=cfg.get("yaw_rate_threshold_deg", 4.0),
        )
        print(
            f"[pipeline] Filtro curve (su dataset completo): "
            f"tenuti {stats['kept']}, scartati {stats['dropped']}"
        )

    # 2) Seleziona il range richiesto fra i frame sopravvissuti
    image_paths = io_data.list_image_paths(images_dir)
    records = io_data.select_range(records_all, image_paths, frame_start, frame_end)
    if len(records) < 1:
        raise RuntimeError(f"Nessun frame nel range [{frame_start}, {frame_end}]")
    print(
        f"[pipeline] Selezionati nel range [{frame_start}, {frame_end}]: {len(records)} frame "
        f"({records[0]['index']}..{records[-1]['index']})"
    )

    # 3) Geodesia UTM locale
    lat0 = sum(r["lat"] for r in records) / len(records)
    lon0 = sum(r["lon"] for r in records) / len(records)
    to_utm, _, utm_crs = make_transformers(lat0, lon0)
    e0, n0 = to_utm.transform(records[0]["lon"], records[0]["lat"])
    ref_origin = (e0, n0)

    # 4) Carica il primo frame per dimensioni; calcola GSD per ogni frame
    first_img = _load_image(records[0]["path"], downscale)
    if first_img is None:
        raise RuntimeError(f"Impossibile leggere {records[0]['path']}")
    h0, w0 = first_img.shape[:2]
    focal_px = K[0, 0] / downscale
    gsds = [gsd_meters_per_pixel(_frame_altitude(r), focal_px) for r in records]
    gsd_canvas = float(np.mean(gsds))
    print(
        f"[pipeline] downscale={downscale} GSD canvas={gsd_canvas:.4f} m/px "
        f"(range {min(gsds):.4f}..{max(gsds):.4f})"
    )

    # 5) Pose iniziali GPS-only
    initial_M = [
        transforms.initial_transform(
            r, to_utm, ref_origin, gsd_canvas, gsd_f, (w0, h0)
        )
        for r, gsd_f in zip(records, gsds)
    ]
    image_sizes = [(w0, h0)] * len(records)

    # 5b) Spacing medio (mediano) tra frame consecutivi -> altezza della striscia per frame
    positions_m = [to_utm.transform(r["lon"], r["lat"]) for r in records]
    distances_m = [
        float(np.hypot(positions_m[i + 1][0] - positions_m[i][0],
                       positions_m[i + 1][1] - positions_m[i][1]))
        for i in range(len(records) - 1)
    ]
    spacing_m = float(np.median(distances_m)) if distances_m else float(h0 * gsd_canvas)
    strip_h_px = max(50, int(round(spacing_m / float(np.mean(gsds)) * 1.15)))
    print(
        f"[pipeline] Spacing GPS medio: {spacing_m:.2f} m | striscia {strip_h_px} px "
        f"(immagine altezza {h0} px)"
    )

    # 6) Calcola canvas
    shifted_M, canvas_size, (off_x, off_y) = mosaic.compute_canvas(initial_M, image_sizes)
    print(f"[pipeline] Canvas: {canvas_size[0]} x {canvas_size[1]}")

    # 6b) Visualizzazione di diagnostica: layout dei frame nel canvas
    layout_img = mosaic.visualize_layout(
        shifted_M, image_sizes, canvas_size, labels=[r["index"] for r in records]
    )
    layout_path = output_jpg.with_name(output_jpg.stem + "_layout.jpg")
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(layout_path), layout_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[pipeline] Layout diagnostico: {layout_path}")

    # 7) Assembla il mosaico
    def loader(i: int):
        return _load_image(records[i]["path"], downscale)

    img, alpha = mosaic.assemble(
        loader,
        shifted_M,
        canvas_size,
        strip_h_px=strip_h_px,
        progress=lambda it: tqdm(it, total=len(records), desc="stitch"),
    )

    # 8) Salva JPG (JPG non supporta alpha: sfondo nero)
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_jpg), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[pipeline] JPG: {output_jpg}")

    # 9) Salva GeoTIFF in WGS84 con canale alpha (trasparenza fuori dal mosaico)
    final_origin = (
        ref_origin[0] + off_x * gsd_canvas,
        ref_origin[1] - off_y * gsd_canvas,
    )
    georef.write_geotiff(
        img,
        canvas_offset_px=(off_x, off_y),
        ref_origin_m=final_origin,
        gsd=gsd_canvas,
        utm_crs=utm_crs,
        output_path=output_tiff,
        alpha_mask=alpha,
    )
    print(f"[pipeline] GeoTIFF (RGBA): {output_tiff}")
