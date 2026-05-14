"""Punto di ingresso. Modifica le costanti in CONFIG e lancia `python main.py`."""
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import features, flight_filter, georef, io_data, matching, mosaic, refine, transforms
from utils.geodesy import gsd_meters_per_pixel, make_transformers

ROOT = Path(__file__).resolve().parent

CONFIG = {
    # --- Percorsi ---
    "calibration_file": ROOT / "data" / "calibration.txt",
    "metadata_file": ROOT / "data" / "metadati.txt",
    "images_dir": ROOT / "immagini" / "immagini_drone" / "immagini_senza_distorsione",
    "output_jpg": ROOT / "output" / "mosaic.jpg",
    "output_tiff": ROOT / "output" / "mosaic.tif",

    # --- Range frame (1-indexed, estremi inclusi). frame_end=None per "fino alla fine". ---
    "frame_start": 1,
    "frame_end": 807,

    # --- Parametri di volo (servono al feature matching per filtrare le coppie di vicini) ---
    "lateral_overlap": 0.81,
    "frontal_overlap": 0.81,

    # --- Performance ---
    "downscale": 2.5,

    # --- Filtro frame in curva ---
    "skip_curves": True,
    "roll_threshold_deg": 10.0,
    "yaw_rate_threshold_deg": 5.0,

    # --- Feature matching + raffinamento globale ---
    "refine_with_features": True,
    "max_features": 3000,             # numero di feature ORB per frame
    "max_neighbors_per_frame": 3,     # massimo n. di coppie matching per frame
    "min_inliers_per_pair": 10,       # sotto questa soglia la coppia viene scartata
    "gps_weight": 0.5,                # quanto pesa il prior GPS rispetto ai vincoli feature
    "refine_max_iter": 30,
}


def _check_inputs(cfg: dict) -> list[str]:
    missing: list[str] = []
    for key in ("calibration_file", "metadata_file"):
        if not cfg[key].is_file():
            missing.append(f"file mancante: {cfg[key]}")
    images_dir: Path = cfg["images_dir"]
    if not images_dir.is_dir():
        missing.append(f"cartella mancante: {images_dir}")
    else:
        if not any(p.suffix.lower() == ".jpg" for p in images_dir.iterdir()):
            missing.append(f"nessun .jpg in {images_dir}")
    return missing


def _validate_range(cfg: dict) -> str | None:
    start, end = cfg["frame_start"], cfg["frame_end"]
    if not isinstance(start, int) or start < 1:
        return f"frame_start non valido: {start}"
    if end is not None and (not isinstance(end, int) or end < start):
        return f"frame_end non valido (deve essere >= {start}): {end}"
    return None


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


def _maybe_refine_poses(
    cfg: dict,
    records: list,
    initial_M: list,
    positions_m_local: list,
    gsd_canvas: float,
    gsds: list,
    image_size: tuple[int, int],
) -> list:
    """Se abilitato, estrae feature, calcola coppie vicini, matcha, raffina. Altrimenti ritorna initial_M."""
    if not cfg.get("refine_with_features", True):
        return initial_M

    w0, h0 = image_size
    downscale: float = cfg.get("downscale", 2.5)
    max_feats: int = cfg.get("max_features", 1500)
    max_neighbors: int = cfg.get("max_neighbors_per_frame", 8)
    min_inliers: int = cfg.get("min_inliers_per_pair", 12)
    gps_weight: float = cfg.get("gps_weight", 0.3)
    refine_iter: int = cfg.get("refine_max_iter", 30)
    lateral_overlap: float = cfg.get("lateral_overlap", 0.5)
    frontal_overlap: float = cfg.get("frontal_overlap", 0.5)

    # 1) Feature extraction: una alla volta per non saturare la RAM
    detector = features.make_detector(max_features=max_feats)
    keypoints_all = []
    descriptors_all = []
    print("[refine] Estrazione feature ORB...")
    for r in tqdm(records, desc="features"):
        img = _load_image(r["path"], downscale)
        if img is None:
            keypoints_all.append([])
            descriptors_all.append(None)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = features.detect(detector, gray)
        keypoints_all.append(kps)
        descriptors_all.append(des)
        del img, gray

    # 2) Coppie di vicini da GPS
    footprint_m = max(w0, h0) * float(np.mean(gsds))
    radius = matching.neighbor_radius_m(footprint_m, lateral_overlap, frontal_overlap)
    pairs = matching.build_neighbor_pairs(
        positions_m_local, radius_m=radius, max_neighbors_per_frame=max_neighbors
    )
    print(f"[refine] Footprint ~{footprint_m:.1f} m, raggio vicini {radius:.1f} m, coppie: {len(pairs)}")

    # 3) Matching + similarity per ogni coppia
    matcher = matching.make_matcher()
    constraints: list[tuple[int, int, np.ndarray, int]] = []
    n_low_inliers = 0
    n_no_homog = 0
    for (i, j) in tqdm(pairs, desc="match"):
        good = matching.match_descriptors(matcher, descriptors_all[i], descriptors_all[j])
        if len(good) < min_inliers:
            n_low_inliers += 1
            continue
        H_ij, n_inl = matching.similarity_from_matches(
            keypoints_all[i], keypoints_all[j], good
        )
        if H_ij is None:
            n_no_homog += 1
            continue
        if n_inl < min_inliers:
            n_low_inliers += 1
            continue
        constraints.append((i, j, H_ij, n_inl))
    print(
        f"[refine] Vincoli validi: {len(constraints)} / {len(pairs)} "
        f"(scartati: {n_low_inliers} per pochi inlier, {n_no_homog} per H_ij None)"
    )

    if len(constraints) < 1:
        print("[refine] Nessun vincolo valido: salto il raffinamento, uso le pose GPS.")
        return initial_M

    # 4) LSQ globale
    print(f"[refine] Least-squares globale: {len(records)*4} parametri, "
          f"{len(records)*8 + len(constraints)*8} residui, gps_weight={gps_weight}")
    refined = refine.refine_poses(
        initial_M,
        constraints,
        image_size=(w0, h0),
        gps_weight=gps_weight,
        max_iter=refine_iter,
        verbose=1,
    )
    return refined


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
    positions_m_local = [(e - e0, n_ - n0) for e, n_ in positions_m]
    distances_m = [
        float(np.hypot(positions_m_local[i + 1][0] - positions_m_local[i][0],
                       positions_m_local[i + 1][1] - positions_m_local[i][1]))
        for i in range(len(records) - 1)
    ]
    spacing_m = float(np.median(distances_m)) if distances_m else float(h0 * gsd_canvas)
    strip_h_px = max(50, int(round(spacing_m / float(np.mean(gsds)) * 1.40)))
    print(
        f"[pipeline] Spacing GPS medio: {spacing_m:.2f} m | striscia {strip_h_px} px "
        f"(immagine altezza {h0} px)"
    )

    # 5c) Feature matching tra vicini + raffinamento globale delle pose
    refined_M = _maybe_refine_poses(cfg, records, initial_M, positions_m_local, gsd_canvas, gsds, (w0, h0))

    # 6) Calcola canvas
    shifted_M, canvas_size, (off_x, off_y) = mosaic.compute_canvas(refined_M, image_sizes)
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


if __name__ == "__main__":
    missing = _check_inputs(CONFIG)
    range_err = _validate_range(CONFIG)
    if missing or range_err:
        print("Errore di configurazione:")
        for m in missing:
            print(f"  - {m}")
        if range_err:
            print(f"  - {range_err}")
        sys.exit(1)
    run_pipeline(CONFIG)
