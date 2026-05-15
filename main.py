"""Punto di ingresso. Modifica le costanti in CONFIG e lancia `python main.py`."""
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from utils import (
    features,
    flight_filter,
    georef,
    image_cache,
    io_data,
    legs as legs_mod,
    matching,
    mosaic,
    odometry,
    overlap as overlap_mod,
    refine,
    transforms,
)
from utils.geodesy import gsd_meters_per_pixel, make_transformers

ROOT = Path(__file__).resolve().parent

CONFIG = {
    # --- Percorsi ---
    "calibration_file": ROOT / "data" / "calibration.txt",
    "metadata_file": ROOT / "data" / "metadati.txt",
    "translations_file": ROOT / "data" / "translations.txt",
    "images_dir": ROOT / "immagini" / "immagini_drone" / "immagini_senza_distorsione",
    "output_jpg": ROOT / "output" / "mosaic.jpg",
    "output_tiff": ROOT / "output" / "mosaic.tif",

    # --- Range frame (1-indexed, estremi inclusi). frame_end=None per "fino alla fine". ---
    "frame_start": 2,
    "frame_end": 89,

    # --- Sorgente dell'overlap: "manual" oppure "auto" (esclusivi). ---
    # "manual" -> usa lateral_overlap / frontal_overlap qui sotto.
    # "auto"   -> dedotti dal pattern GPS + footprint (ignora i valori manuali).
    "overlap_source": "auto",
    "lateral_overlap": 0.745,
    "frontal_overlap": 0.815,

    # --- Performance ---
    "downscale": 2.5,
    "jpeg_quality_cache": 85,

    # --- Strategia composizione mosaico ---
    # "first":  pixel dal primo frame in ordine temporale (seam "casuali").
    # "center": pixel dal frame in cui e' piu' nadirale (zona meno distorta).
    "blend_mode": "center",

    # --- Filtro frame in curva (vengono marcati ma NON scartati: fanno da bridge fra leg) ---
    "roll_threshold_deg": 10.0,
    "yaw_rate_threshold_deg": 5.0,

    # --- Feature matching + raffinamento globale (sempre attivo) ---
    "max_features": 3000,
    "min_inliers_per_pair": 10,
    # Matching cross-leg: quanti vicini al massimo (di leg diversi, entro radius
    # da lateral_overlap) cercare per ogni frame del mosaico. Aggancia direttamente
    # leg adiacenti senza dover passare per i bridge curva, che sono visualmente sgualciti.
    "max_cross_leg_neighbors": 6,
    # Pesi LSQ: GPS = anchor assoluto debole (rumoroso a ~5 m), VO = inter-frame
    # relativo forte (precisa a ~10 cm, ma drifta nel lungo periodo). Le feature
    # restano col proprio peso (sqrt(n_inliers)) per il fine-tuning.
    "gps_weight": 0.5,
    "vo_weight": 5.0,
    "refine_max_iter": 400,
}


def _check_inputs(cfg: dict) -> list[str]:
    missing: list[str] = []
    for key in ("calibration_file", "metadata_file", "translations_file"):
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


def _validate_overlap_source(cfg: dict) -> str | None:
    src = cfg.get("overlap_source")
    if src not in ("manual", "auto"):
        return (
            f"overlap_source non valido: {src!r}. "
            f"Deve essere 'manual' oppure 'auto'."
        )
    return None


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
    jpeg_q: int = cfg.get("jpeg_quality_cache", 85)
    blend_mode: str = cfg.get("blend_mode", "first")

    # 1) Calibrazione + metadati + delta VO; marca le curve (senza scartarle).
    K = io_data.load_camera_matrix(cfg["calibration_file"])
    records_all = io_data.load_metadata(cfg["metadata_file"])
    vo_deltas = odometry.load_vo_deltas(cfg["translations_file"])
    if len(vo_deltas) != len(records_all):
        raise RuntimeError(
            f"VO/metadata mismatch: {len(vo_deltas)} delta vs {len(records_all)} record"
        )
    curve_stats = flight_filter.mark_curves(
        records_all,
        roll_threshold_deg=cfg.get("roll_threshold_deg", 8.0),
        yaw_rate_threshold_deg=cfg.get("yaw_rate_threshold_deg", 4.0),
    )
    print(
        f"[pipeline] Curve marcate sul dataset: {curve_stats['curve']} "
        f"(dritti: {curve_stats['straight']})"
    )

    # 2) Selezione range
    image_paths = io_data.list_image_paths(images_dir)
    records = io_data.select_range(records_all, image_paths, frame_start, frame_end)
    if len(records) < 1:
        raise RuntimeError(f"Nessun frame nel range [{frame_start}, {frame_end}]")
    n_curve = sum(1 for r in records if r["is_curve"])
    n_straight = len(records) - n_curve
    print(
        f"[pipeline] Range [{frame_start}, {frame_end}]: {len(records)} frame "
        f"({n_straight} dritti + {n_curve} curva) "
        f"({records[0]['index']}..{records[-1]['index']})"
    )

    # 3) Cache RAM: una sola lettura da disco, downscale e riencode JPEG.
    cache = image_cache.FrameCache(
        paths=[r["path"] for r in records],
        downscale=downscale,
        jpeg_quality=jpeg_q,
    )
    w0, h0 = cache.image_size

    # 4) Geodesia UTM locale + posizioni in metri
    lat0 = sum(r["lat"] for r in records) / len(records)
    lon0 = sum(r["lon"] for r in records) / len(records)
    to_utm, _, utm_crs = make_transformers(lat0, lon0)
    e0, n0 = to_utm.transform(records[0]["lon"], records[0]["lat"])
    ref_origin = (e0, n0)
    positions_m = [to_utm.transform(r["lon"], r["lat"]) for r in records]
    positions_m_local = [(e - e0, n_ - n0) for e, n_ in positions_m]

    # 5) GSD per frame (focal a risoluzione dopo downscale)
    focal_px = K[0, 0] / downscale
    gsds = [gsd_meters_per_pixel(_frame_altitude(r), focal_px) for r in records]
    gsd_canvas = float(np.mean(gsds))
    print(
        f"[pipeline] downscale={downscale} GSD canvas={gsd_canvas:.4f} m/px "
        f"(range {min(gsds):.4f}..{max(gsds):.4f})"
    )

    # 6) Raggruppa in leg (passate dritte) con bridge curva ai confini
    legs = legs_mod.group_into_legs(records)
    print(f"[pipeline] Leg identificati: {len(legs)}")
    for k, leg in enumerate(legs):
        print(
            f"  leg #{k}: {len(leg['frames'])} frame dritti | "
            f"bridge prev={len(leg['bridge_prev'])}, next={len(leg['bridge_next'])}"
        )

    # 7) Risolvi sorgente overlap (manual XOR auto)
    src = overlap_mod.resolve_overlap_source(cfg)
    if src == "manual":
        lateral_ov = float(cfg["lateral_overlap"])
        frontal_ov = float(cfg["frontal_overlap"])
        print(
            f"[pipeline] Overlap manuale: lat={lateral_ov:.3f} front={frontal_ov:.3f}"
        )
    else:
        lateral_ov, frontal_ov = overlap_mod.compute_auto_overlap(
            legs, positions_m_local, (w0, h0), gsds
        )
        print(
            f"[pipeline] Overlap auto: lat={lateral_ov:.3f} front={frontal_ov:.3f}"
        )

    # 8) Sottocampiona ciascun leg in base al frontal_overlap target
    stitch_indices_per_leg: list[list[int]] = []
    for leg in legs:
        frames = leg["frames"]
        leg_positions = [positions_m_local[i] for i in frames]
        local_kept = legs_mod.subsample_leg_by_overlap(
            leg_positions,
            image_height_px=h0,
            gsd_m_per_px=gsd_canvas,
            target_frontal_overlap=frontal_ov,
        )
        stitch_indices_per_leg.append([frames[k] for k in local_kept])
    total_stitch = sum(len(s) for s in stitch_indices_per_leg)
    print(
        f"[pipeline] Frame nel mosaico dopo sottocampionamento: "
        f"{total_stitch}/{n_straight}"
    )

    # 9) Insieme dei frame coinvolti nella stima pose: frame mosaico + tutti i bridge
    pose_set: set[int] = set()
    for s in stitch_indices_per_leg:
        pose_set.update(s)
    for leg in legs:
        pose_set.update(leg["bridge_prev"])
        pose_set.update(leg["bridge_next"])
    pose_indices = sorted(pose_set)
    record_to_pose = {idx: k for k, idx in enumerate(pose_indices)}

    # 10) Pose iniziali GPS-only per i frame coinvolti
    initial_M = [
        transforms.initial_transform(
            records[i], to_utm, ref_origin, gsd_canvas, gsds[i], (w0, h0)
        )
        for i in pose_indices
    ]

    # 11) Feature ORB sui soli frame coinvolti, leggendo dalla cache
    detector = features.make_detector(max_features=cfg.get("max_features", 3000))
    keypoints_all: list = []
    descriptors_all: list = []
    print("[refine] Estrazione feature ORB...")
    for i in tqdm(pose_indices, desc="features"):
        img = cache.get(i)
        if img is None:
            keypoints_all.append([])
            descriptors_all.append(None)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, des = features.detect(detector, gray)
        keypoints_all.append(kps)
        descriptors_all.append(des)

    # 12) Coppie da matchare: catena intra-leg + aggancio diretto cross-leg
    pairs_intra = matching.build_leg_pairs(legs, stitch_indices_per_leg)
    cross_radius = matching.cross_leg_radius_m(
        image_width_px=w0,
        gsd_m_per_px=gsd_canvas,
        lateral_overlap=lateral_ov,
    )
    pairs_cross = matching.build_cross_leg_pairs(
        stitch_indices_per_leg,
        positions_m_local,
        radius_m=cross_radius,
        max_neighbors_per_frame=cfg.get("max_cross_leg_neighbors", 4),
    )
    pairs_record = sorted(set(pairs_intra) | set(pairs_cross))
    pairs = [(record_to_pose[a], record_to_pose[b]) for a, b in pairs_record]
    print(
        f"[refine] Coppie: intra-leg {len(pairs_intra)} + cross-leg {len(pairs_cross)} "
        f"-> totale unico {len(pairs)} (radius cross-leg {cross_radius:.1f} m)"
    )

    # 12b) Vincoli VO inter-frame: per ogni coppia consecutiva in pose_indices,
    # somma cumulativa dei delta VO da records[a] a records[b], convertita in pixel canvas.
    vo_inter_constraints: list[tuple[int, int, float, float]] = []
    for k in range(len(pose_indices) - 1):
        a = pose_indices[k]
        b = pose_indices[k + 1]
        va = records[a]["vo_idx"]
        vb = records[b]["vo_idx"]
        de_m, dn_m = odometry.cumulative_delta_m(vo_deltas, va, vb)
        dx_px = de_m / gsd_canvas
        dy_px = -dn_m / gsd_canvas  # canvas-y cresce verso sud
        vo_inter_constraints.append(
            (record_to_pose[a], record_to_pose[b], dx_px, dy_px)
        )
    print(f"[refine] Vincoli VO inter-frame: {len(vo_inter_constraints)}")

    # 13) Matching + similarity per ogni coppia
    matcher = matching.make_matcher()
    constraints: list[tuple[int, int, np.ndarray, int]] = []
    min_inliers = cfg.get("min_inliers_per_pair", 10)
    n_low_inliers = 0
    n_no_homog = 0
    for (a, b) in tqdm(pairs, desc="match"):
        good = matching.match_descriptors(
            matcher, descriptors_all[a], descriptors_all[b]
        )
        if len(good) < min_inliers:
            n_low_inliers += 1
            continue
        H_ab, n_inl = matching.similarity_from_matches(
            keypoints_all[a], keypoints_all[b], good
        )
        if H_ab is None:
            n_no_homog += 1
            continue
        if n_inl < min_inliers:
            n_low_inliers += 1
            continue
        constraints.append((a, b, H_ab, n_inl))
    print(
        f"[refine] Vincoli validi: {len(constraints)}/{len(pairs)} "
        f"(scartati: {n_low_inliers} per pochi inlier, {n_no_homog} per H None)"
    )

    # 14) Refinement globale LSQ (sempre attivo): anchor GPS + match feature + VO inter-frame
    gps_w = cfg.get("gps_weight", 0.2)
    vo_w = cfg.get("vo_weight", 5.0)
    n_res = (
        len(pose_indices) * 8
        + len(constraints) * 8
        + len(vo_inter_constraints) * 2
    )
    print(
        f"[refine] LSQ globale: {len(pose_indices)*4} parametri, {n_res} residui, "
        f"gps_weight={gps_w}, vo_weight={vo_w}"
    )
    refined_all = refine.refine_poses(
        initial_M,
        constraints,
        image_size=(w0, h0),
        gps_weight=gps_w,
        vo_constraints=vo_inter_constraints,
        vo_weight=vo_w,
        max_iter=cfg.get("refine_max_iter", 30),
        verbose=1,
    )

    # 15) Estrai solo le pose dei frame del mosaico (no bridge), in ordine temporale
    stitch_indices: list[int] = sorted(
        idx for ids in stitch_indices_per_leg for idx in ids
    )
    stitch_M = [refined_all[record_to_pose[i]] for i in stitch_indices]
    image_sizes = [(w0, h0)] * len(stitch_indices)

    # 15b) Marca primo/ultimo frame di ciascun leg per estendere la striscia
    # (evita rettangoli neri ai confini fra leg)
    stitch_pos = {idx: pos for pos, idx in enumerate(stitch_indices)}
    leg_first_positions: set[int] = set()
    leg_last_positions: set[int] = set()
    for ids in stitch_indices_per_leg:
        if ids:
            leg_first_positions.add(stitch_pos[ids[0]])
            leg_last_positions.add(stitch_pos[ids[-1]])

    # 16) Canvas sui soli frame del mosaico
    shifted_M, canvas_size, (off_x, off_y) = mosaic.compute_canvas(
        stitch_M, image_sizes
    )
    print(f"[pipeline] Canvas: {canvas_size[0]} x {canvas_size[1]}")

    # 17) Spacing per la striscia (sui frame del mosaico)
    distances_m = [
        float(
            np.hypot(
                positions_m_local[stitch_indices[k + 1]][0]
                - positions_m_local[stitch_indices[k]][0],
                positions_m_local[stitch_indices[k + 1]][1]
                - positions_m_local[stitch_indices[k]][1],
            )
        )
        for k in range(len(stitch_indices) - 1)
    ]
    spacing_m = (
        float(np.median(distances_m)) if distances_m else float(h0 * gsd_canvas)
    )
    strip_h_px = max(50, int(round(spacing_m / float(np.mean(gsds)) * 1.40)))
    print(
        f"[pipeline] Spacing GPS mediano (kept): {spacing_m:.2f} m | "
        f"striscia {strip_h_px} px (altezza immagine {h0} px)"
    )

    # 18) Layout diagnostico (etichette = frame index del file)
    layout_img = mosaic.visualize_layout(
        shifted_M,
        image_sizes,
        canvas_size,
        labels=[records[i]["index"] for i in stitch_indices],
    )
    layout_path = output_jpg.with_name(output_jpg.stem + "_layout.jpg")
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(layout_path), layout_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"[pipeline] Layout diagnostico: {layout_path}")

    # 19) Assembla mosaico dai frame in cache
    def loader(k: int):
        return cache.get(stitch_indices[k])

    print(f"[pipeline] Composizione: blend_mode={blend_mode}")
    img, alpha = mosaic.assemble(
        loader,
        shifted_M,
        canvas_size,
        strip_h_px=strip_h_px,
        leg_first_indices=leg_first_positions,
        leg_last_indices=leg_last_positions,
        blend_mode=blend_mode,
        progress=lambda it: tqdm(it, total=len(stitch_indices), desc="stitch"),
    )

    # 20) Salva JPG
    output_jpg.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_jpg), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[pipeline] JPG: {output_jpg}")

    # 21) Salva GeoTIFF con alpha
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
    overlap_err = _validate_overlap_source(CONFIG)
    if missing or range_err or overlap_err:
        print("Errore di configurazione:")
        for m in missing:
            print(f"  - {m}")
        if range_err:
            print(f"  - {range_err}")
        if overlap_err:
            print(f"  - {overlap_err}")
        sys.exit(1)
    run_pipeline(CONFIG)
