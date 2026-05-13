"""Orchestrazione end-to-end della pipeline VO + stitching + ortomosaico.

Ingresso: una `Config` popolata da `main.py`.
Uscita (cartella `output/`):
    - `mosaic.png` + `mosaic.pgw` + `mosaic.prj`   -> ortomosaico georeferenziato (UTM)
    - `path_vo_vs_gps.html`                          -> grafico Plotly con polilinee sovrapposte
    - `chunks/pass_<k>.png`                          -> debug, una immagine per passata
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .compose import compose_chunks, refine_chunks_inter_pass
from .config import Config
from .features import detect_orb_grid  # noqa: F401  (riesporto utile)
from .geo import latlon_to_utm, apply_similarity, umeyama
from .io_calibration import load_calibration
from .io_metadata import FrameMeta, load_metadata, load_scales
from .ortho import write_georeferenced_mosaic
from .passes import segment_passes
from .stitch_chunk import build_chunk_for_pass, Chunk
from .vo import run_vo


def _resolve_undistorted_paths(metas: list[FrameMeta], images_dir: Path) -> list[Path]:
    """Mappa ogni metadato al corrispondente file in `immagini_senza_distorsione/`.

    Si basa sul basename: i file undistorti hanno lo stesso nome dei raw.
    """
    out: list[Path] = []
    missing: list[str] = []
    for m in metas:
        name = Path(m.source).name
        p = images_dir / name
        if not p.exists():
            missing.append(name)
            continue
        out.append(p)
    if missing:
        print(f"[pipeline] ATTENZIONE: {len(missing)} immagini mancanti in {images_dir} "
              f"(es. {missing[:3]})", file=sys.stderr)
    return out


def _slice_range(cfg: Config, n: int) -> tuple[int, int]:
    """Calcola gli indici [start, end) dato cfg e dimensione totale `n`."""
    if cfg.start == 0 and cfg.end == 0:
        return 0, n
    if cfg.end == 0:
        return cfg.start, n
    return cfg.start, min(cfg.end, n)


def _estimate_agl(metas: list[FrameMeta], assumed: float | None) -> float:
    """Determina l'AGL costante da usare per la rettifica dei frame.

    - se `cfg.assumed_agl_m` e' impostato, lo usa
    - altrimenti default a 80 m (mission cruise tipica DJI/UgCS)
    """
    if assumed is not None and assumed > 0:
        return float(assumed)
    # In assenza di DTM si usa un valore di default ragionevole.
    return 80.0


def _read_grayscale_downscaled(path: Path, downscale: float) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    if downscale and downscale > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w / downscale), int(h / downscale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def run(cfg: Config) -> None:
    t0 = time.perf_counter()
    out_dir = cfg.output_dir
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento dati ---------------------------------------------------
    K, _P, _dist = load_calibration(cfg.data_dir)
    metas = load_metadata(cfg.metadati_file)
    scales_full = load_scales(cfg.scales_file) if cfg.scales_file.exists() else None

    paths = _resolve_undistorted_paths(metas, cfg.images_dir)
    if len(paths) != len(metas):
        # Alcuni metadati senza immagine: tronchiamo a quanti combaciano in ordine
        metas = metas[: len(paths)]

    n_total = len(paths)
    s, e = _slice_range(cfg, n_total)
    paths = paths[s:e]
    metas = metas[s:e]
    if scales_full is not None:
        # scales[i] = distanza tra frame i-1 e i (m). Per il sotto-range a partire
        # da `s` la "scala iniziale" non e' usata (i=0 nello slice).
        scales = scales_full[s:e]
    else:
        scales = None
    print(f"[pipeline] frame in lavorazione: {len(paths)} (range {s}:{e})", flush=True)
    if not paths:
        print("[pipeline] nessun frame da processare. Fine.")
        return

    # 2. GPS -> UTM ---------------------------------------------------------
    lats = np.array([m.lat for m in metas])
    lons = np.array([m.lon for m in metas])
    yaws = np.array([m.flight_yaw for m in metas])
    gps_utm, epsg = latlon_to_utm(lats, lons)
    print(f"[pipeline] EPSG UTM: {epsg}; bbox UTM "
          f"E[{gps_utm[:,0].min():.1f}..{gps_utm[:,0].max():.1f}] "
          f"N[{gps_utm[:,1].min():.1f}..{gps_utm[:,1].max():.1f}]", flush=True)

    # 3. Segmentazione passate ---------------------------------------------
    passes = segment_passes(
        yaws,
        turn_thresh_deg=cfg.pass_yaw_turn_threshold_deg,
        min_pass_len=cfg.pass_min_len,
    )
    print(f"[pipeline] passate rilevate: {len(passes)} -> {passes}", flush=True)

    # 4. AGL ----------------------------------------------------------------
    agl = _estimate_agl(metas, cfg.assumed_agl_m)
    gsd_native = float(agl) / float(K[0, 0])
    print(f"[pipeline] AGL stimata: {agl:.1f} m -> GSD nativa: {gsd_native*100:.2f} cm/px; "
          f"canvas GSD: {cfg.canvas_resolution_m_per_px*100:.2f} cm/px", flush=True)

    # 5. Costruzione chunk per passata --------------------------------------
    chunks: list[Chunk] = []
    for k, (ps, pe) in enumerate(passes):
        sub_paths = paths[ps:pe]
        sub_yaws = yaws[ps:pe].tolist()
        sub_utm = gps_utm[ps:pe]
        save = chunk_dir / f"pass_{k:02d}.png" if cfg.debug.save_chunks else None
        chunk = build_chunk_for_pass(
            sub_paths, sub_yaws, sub_utm, agl_m=agl, K=K,
            pass_index=k, frame_indices=list(range(ps, pe)),
            canvas_gsd_m_per_px=cfg.canvas_resolution_m_per_px,
            save_path=save,
        )
        if chunk is None:
            print(f"[pipeline] chunk passata {k} non costruito (vuoto / patologico)")
            continue
        chunks.append(chunk)
        print(f"[pipeline] chunk passata {k}: {chunk.image.shape[1]}x{chunk.image.shape[0]} px "
              f"({len(sub_paths)} frame)", flush=True)

    if not chunks:
        print("[pipeline] nessun chunk costruito; abort.")
        return

    # 6. Refinement inter-passata (GPS-aware -> match feature in sovrapposizione) ---
    chunks = refine_chunks_inter_pass(chunks, n_features_total=2500, n_grid=4,
                                      lowe_ratio=cfg.lowe_ratio,
                                      ransac_thresh=cfg.ransac_thresh_px,
                                      verbose=cfg.debug.verbose)

    # 7. Composizione globale ----------------------------------------------
    mosaic = compose_chunks(chunks, canvas_gsd_m_per_px=cfg.canvas_resolution_m_per_px)
    print(f"[pipeline] mosaico globale: {mosaic.image.shape[1]}x{mosaic.image.shape[0]} px", flush=True)

    # 8. Scrittura ortomosaico georeferenziato ------------------------------
    written = write_georeferenced_mosaic(mosaic, out_dir / "mosaic", epsg_utm=epsg)
    print(f"[pipeline] scritto mosaico: {written['image'].name} (+ .pgw + .prj)", flush=True)

    # 9. VO + plot VO vs GPS ------------------------------------------------
    print(f"[pipeline] VO su {len(paths)} frame...", flush=True)
    grays = [_read_grayscale_downscaled(p, cfg.image_downscale_for_vo) for p in paths]
    vo = run_vo(
        grays, K=K,
        scales=scales,
        n_grid=cfg.n_grid,
        n_features_total=cfg.n_features_total,
        lowe_ratio=cfg.lowe_ratio,
        min_matches=cfg.min_matches,
        essential_prob=cfg.essential_prob,
        essential_thresh=cfg.essential_thresh_px,
        use_imu_scales=cfg.use_imu_scales,
        verbose=cfg.debug.verbose,
    )
    print(f"[pipeline] VO completata. Frame skippati: {len(vo.skipped)}", flush=True)

    # NB: i grays sono stati scalati da downscale ma le scales IMU sono in metri,
    # quindi il VO e' in metri (modulo errori di scala IMU). Le coordinate VO
    # sono "world" in un sistema arbitrariamente orientato; le allineiamo al
    # GPS via Umeyama (similarity 2D).
    vo_xy = vo.path[:, :2]                 # primi due assi VO (X, Y) in metri
    gps_xy_local = gps_utm - gps_utm[0]    # ENU locale, origine sul primo frame
    s_um, R_um, t_um = umeyama(vo_xy, gps_xy_local, with_scale=True)
    vo_aligned = apply_similarity(s_um, R_um, t_um, vo_xy)
    print(f"[pipeline] Umeyama VO->GPS: scala={s_um:.4f}, det(R)={np.linalg.det(R_um):+.3f}, t={t_um}",
          flush=True)

    # Salva il plot via il modulo visualization.plot_one (gia' supporta GPS overlay)
    from visualization.plot_one import visualize_path_2d  # import locale per evitare side effect
    plot_path = out_dir / "path_vo_vs_gps.html"
    # `visualize_path_2d` si aspetta path 2D/3D. Lavoriamo in 2D ENU.
    vo_3d = np.column_stack([vo_aligned, np.zeros(len(vo_aligned))])
    visualize_path_2d(vo_3d, "VO vs GPS (UTM-locale, m)", file_out=str(plot_path),
                      gps_path=gps_xy_local)
    print(f"[pipeline] plot scritto: {plot_path}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[pipeline] FATTO in {elapsed:.1f} s.", flush=True)
