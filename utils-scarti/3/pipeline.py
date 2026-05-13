"""Orchestratore della pipeline: stitching + visual odometry.

Lo stitching ha due backend selezionabili via `cfg.mosaic_method`:
  - "ortho" (default): ortomosaico geo-riferito da GPS + gimbal yaw, con
    AGL auto-calibrata. Drift zero per costruzione, output metrico.
  - "scans": `cv2.Stitcher_SCANS`, pipeline solo-pixel di OpenCV. Più
    permissiva ma soggetta a drift su sequenze lunghe.

La VO è indipendente: estrae feature ORB su griglia, matcha pairwise e
stima R, t con la matrice essenziale.
"""
import csv

import cv2
import numpy as np

from .config import Config
from .io_calibration import (
    load_calibration, load_images, load_gps_path,
    load_frame_geometry,
)
from .features import FeatureExtractor
from .matching import make_flann_matcher, match_pairs
from .mosaic import render_mosaic
from .ortho import render_ortho, enu_to_canvas
from .visual_odometry import estimate_pose, Trajectory
from .timing import timed, set_log_path
from . import bootstrap, debug


@timed
def run(cfg: Config):
    """Esegue la pipeline completa: stitching + VO + output su disco.

    Ritorna (mosaic, trajectory).
    """
    out_dir = cfg.resolve(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_log_path(out_dir / f"timings_{cfg.start}_to_{cfg.end}.log")

    cv2.ocl.setUseOpenCL(cfg.use_opencl)

    bootstrap.verify_or_build(cfg)

    K_full, _ = load_calibration(cfg.resolve(cfg.calibration_file))
    images, _paths = load_images(
        cfg.resolve(cfg.images_glob),
        resize_factor=cfg.resize_factor,
        start=cfg.start, end=cfg.end,
    )
    # K letta da disco è full-res; le immagini sono ridotte di resize_factor.
    # Scaliamo una sola volta qui: la VO ha bisogno della K coerente con i pixel
    # che vede (altrimenti findEssentialMat normalizza male e recoverPose può
    # invertire il segno di t su scene planari nadir → flip della traiettoria).
    K = K_full.copy()
    if cfg.resize_factor != 1.0:
        K[:2] /= cfg.resize_factor
    if len(images) < 2:
        raise RuntimeError("Servono almeno 2 immagini per lo stitching.")
    print(f"Caricate {len(images)} immagini.")

    geometry = load_frame_geometry(
        cfg.resolve("data/metadati.txt"),
        start=cfg.start, end=cfg.end,
    )
    if len(geometry.positions_enu) != len(images):
        raise RuntimeError(
            f"Geometry/images disallineati: {len(geometry.positions_enu)} vs {len(images)}."
        )
    # Scale per-frame derivate dal GPS: modulo dello spostamento ENU tra frame
    # i-1 e i. Sostituisce data/scales.txt (basato su FlightSpeed × delta_time
    # con timestamp EXIF arrotondati al secondo, che produceva picchi a ~2× il
    # valore reale ogni ~16 frame e drift cumulativo nella VO).
    displacements = np.zeros(len(images))
    displacements[1:] = np.linalg.norm(np.diff(geometry.positions_enu, axis=0), axis=1)

    trajectory = _run_visual_odometry(images, K, displacements, geometry, cfg)

    mosaic, ortho_result = _render_mosaic(images, K, geometry, cfg)

    mosaic_path = out_dir / f"mosaic_{cfg.start}_to_{cfg.end}.jpg"
    cv2.imwrite(str(mosaic_path), mosaic)
    print(f"Mosaico salvato in {mosaic_path}")

    vo_path_enu = _vo_to_enu(trajectory.path, cfg)

    if ortho_result is not None:
        overlay_path = out_dir / f"mosaic_with_trajectory_{cfg.start}_to_{cfg.end}.jpg"
        _save_mosaic_with_trajectories(
            mosaic, ortho_result, geometry, vo_path_enu, overlay_path,
        )

    try:
        gps_path = load_gps_path(
            cfg.resolve("data/metadati.txt"),
            start=cfg.start, end=cfg.end,
        )
    except Exception as e:
        print(f"GPS non caricato ({e}); proseguo senza overlay.")
        gps_path = None

    try:
        from visualization import plot_one
        traj_path = out_dir / f"trajectory_{cfg.start}_to_{cfg.end}.html"
        plot_one.visualize_path_2d(
            vo_path_enu, "Percorso 2D stimato (VO vs GPS)",
            file_out=str(traj_path),
            gps_path=gps_path,
        )
        print(f"Traiettoria salvata in {traj_path}")
    except Exception as e:
        print(f"Plot traiettoria saltato: {e}")

    return mosaic, trajectory


@timed
def _vo_to_enu(vo_path, cfg: Config):
    """Ruota la traiettoria VO dal frame camera-0 al riferimento mondo ENU.

    La VO accumula pose nel frame della camera al frame 0; per nadir+FollowYaw
    questo frame è ruotato di `yaw_0` (compass, +cw da nord) rispetto al mondo:
      x_cam(0) world = compass(yaw_0 + 90°) = ( cos yaw_0, -sin yaw_0)
      y_cam(0) world = compass(yaw_0 + 180°) = (-sin yaw_0, -cos yaw_0)
    Per un punto (a, b, c) nel frame cam-0:
      E = a·cos yaw_0 - b·sin yaw_0
      N = -a·sin yaw_0 - b·cos yaw_0
    z_cam punta verso il basso (z_cam = -Up), quindi U = -c.
    """
    import numpy as np
    path = np.asarray(vo_path, dtype=float)
    if len(path) == 0:
        return path
    geom = load_frame_geometry(
        cfg.resolve("data/metadati.txt"),
        start=cfg.start, end=cfg.end,
    )
    if len(geom.yaw_deg) == 0:
        return path
    yaw0 = float(geom.yaw_deg[0])
    c = np.cos(np.radians(yaw0))
    s = np.sin(np.radians(yaw0))
    a, b, z = path[:, 0], path[:, 1], path[:, 2]
    E = a * c - b * s
    N = -a * s - b * c
    U = -z
    return np.stack([E, N, U], axis=1)


@timed
def _render_mosaic(images, K, geometry, cfg: Config):
    """Compone il mosaico col backend selezionato da cfg.mosaic_method.

    Ritorna `(image, ortho_result_or_None)`. `ortho_result` è popolato solo per
    il backend "ortho" e contiene i parametri di geo-referenziazione (gsd,
    east_min, north_max) necessari per proiettare ENU → pixel canvas — usato
    per disegnare le traiettorie sul mosaico.
    """
    method = cfg.mosaic_method
    if method == "scans":
        print(f"Stitching di {len(images)} immagini con cv2.Stitcher SCANS...")
        return render_mosaic(images, confidence_thresh=cfg.stitcher_confidence_thresh), None

    if method == "ortho":
        print(f"Ortomosaico geo-riferito di {len(images)} immagini...")
        result = render_ortho(
            images, geometry, K,
            agl=cfg.ortho_agl_m,
            gsd=cfg.ortho_gsd,
        )
        return result.image, result

    raise ValueError(f"cfg.mosaic_method sconosciuto: {method!r}")


def _save_mosaic_with_trajectories(mosaic, ortho_result, geometry, vo_path_enu, out_path):
    """Disegna GPS (rosso tratteggiato) e VO (blu) sull'ortomosaico e salva.

    GPS è già nel frame ENU mondo: proiezione diretta via `enu_to_canvas`.
    VO arriva da `_vo_to_enu` centrato in (0, 0, 0) (origine = posa cam-0): va
    ancorato traslandolo sulla posizione GPS del primo frame, altrimenti finisce
    fuori canvas (il canvas copre l'estensione del volo, non l'origine ENU).

    Convenzione colori: blu/rosso = stessi della HTML in `plot_one.visualize_path_2d`
    (riconoscibili tra le due viste).
    """
    overlay = mosaic.copy()
    h, w = overlay.shape[:2]
    thickness = max(2, int(round(max(h, w) / 800)))

    gps_canvas = enu_to_canvas(geometry.positions_enu, ortho_result)
    gps_pts = np.round(gps_canvas).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(
        overlay, [gps_pts], isClosed=False, color=(0, 0, 255),
        thickness=thickness, lineType=cv2.LINE_AA,
    )

    if vo_path_enu is not None and len(vo_path_enu) >= 2:
        vo_world = np.asarray(vo_path_enu, dtype=np.float64).copy()
        vo_world[:, 0] += geometry.positions_enu[0, 0]
        vo_world[:, 1] += geometry.positions_enu[0, 1]
        vo_canvas = enu_to_canvas(vo_world, ortho_result)
        vo_pts = np.round(vo_canvas).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            overlay, [vo_pts], isClosed=False, color=(255, 0, 0),
            thickness=thickness, lineType=cv2.LINE_AA,
        )

    # Marker start (verde) e end (giallo) sulla GPS — orientamento immediato.
    radius = thickness * 4
    cv2.circle(overlay, tuple(np.round(gps_canvas[0]).astype(int)), radius,
               (0, 255, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(overlay, tuple(np.round(gps_canvas[-1]).astype(int)), radius,
               (0, 255, 255), -1, lineType=cv2.LINE_AA)

    cv2.imwrite(str(out_path), overlay)
    print(f"Mosaico con traiettorie (GPS rosso, VO blu) salvato in {out_path}")


@timed
def _run_visual_odometry(images, K, displacements, geometry, cfg: Config) -> Trajectory:
    """Estrae feature ORB su griglia, matcha pairwise e stima la traiettoria 2D.

    `displacements[i]` è il modulo (m) dello spostamento GPS tra frame i-1 e i;
    `displacements[0]` è 0 e non viene mai usato (il loop parte da i=1). Fornisce
    la scala metrica al versore `t` restituito da `recoverPose`.

    `geometry` serve solo come riferimento esterno per disambiguare il segno di
    `t`: su scena planare nadir con bassa parallasse `recoverPose` può sceglierne
    arbitrariamente uno dei due, e una volta che la VO si rovescia non si recupera.
    """
    extractor = FeatureExtractor(
        n_grid=cfg.n_grid,
        n_features_total=cfg.n_features_total,
        overlap=cfg.quadrant_overlap,
    )
    matcher = make_flann_matcher()
    trajectory = Trajectory()

    feat_dir = cfg.resolve(cfg.debug.features_dir)
    if cfg.debug.save_features:
        feat_dir.mkdir(parents=True, exist_ok=True)

    prev_gray = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
    prev_kps, prev_des = extractor.detect(prev_gray)
    if cfg.debug.save_features:
        debug.save_features_image(
            images[0], prev_kps,
            feat_dir / f"feature_img_{cfg.start}.jpg",
        )

    n_flips = 0
    n_fallbacks = 0
    diag_rows = []
    for i in range(1, len(images)):
        curr_gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        curr_kps, curr_des = extractor.detect(curr_gray)

        good, q_prev, q_curr = match_pairs(
            matcher, prev_kps, prev_des, curr_kps, curr_des,
            ratio=cfg.lowe_ratio,
            min_good=cfg.min_good_matches,
        )
        if good is None:
            print(f"VO: match insufficienti tra {i - 1} e {i}, traiettoria invariata.")
            prev_kps, prev_des = curr_kps, curr_des
            diag_rows.append({'i': i, 'status': 'no_matches'})
            continue

        expected_R = _expected_R_from_yaw(
            float(geometry.yaw_deg[i - 1]), float(geometry.yaw_deg[i]),
        )
        R, t = estimate_pose(
            q_prev, q_curr, K,
            expected_R=expected_R,
            ransac_thr=cfg.essential_ransac_thr,
            prob=cfg.essential_ransac_prob,
        )
        if R is not None:
            t, flags = _disambiguate_t_sign(R, t, geometry, i)
            if flags["flipped"]:
                n_flips += 1
            if flags["planar_fallback"]:
                n_fallbacks += 1
            trajectory.update(R, t * float(displacements[i]))
            diag_rows.append(_vo_diagnostics(R, t, geometry, displacements, i, flags))
        else:
            print(f"VO: posa non stimabile al frame {i}, traiettoria invariata.")
            diag_rows.append({'i': i, 'status': 'no_pose'})

        if cfg.debug.save_features:
            debug.save_features_image(
                images[i], curr_kps,
                feat_dir / f"feature_img_{cfg.start + i}.jpg",
            )

        prev_kps, prev_des = curr_kps, curr_des

        if i % 20 == 0 or i == len(images) - 1:
            print(f"  VO {i}/{len(images) - 1}")

    if n_flips:
        print(f"VO: segno di t invertito su {n_flips}/{len(images) - 1} frame.")
    if n_fallbacks:
        print(
            f"VO: t sostituito con direzione GPS su {n_fallbacks}/{len(images) - 1} "
            f"frame degeneri (scena planare, ‖t_xy‖ < soglia)."
        )
    _write_vo_diagnostics(diag_rows, cfg)
    _print_vo_summary(diag_rows)
    return trajectory


def _vo_diagnostics(R, t, geometry, displacements, i, flags):
    """Raccoglie metriche per-frame della VO (vedi `_write_vo_diagnostics`)."""
    t_arr = np.asarray(t).ravel()
    delta = geometry.positions_enu[i] - geometry.positions_enu[i - 1]
    yaw_prev = float(np.radians(geometry.yaw_deg[i - 1]))
    cos_y, sin_y = np.cos(yaw_prev), np.sin(yaw_prev)
    expected = np.array([
         delta[0] * cos_y - delta[1] * sin_y,
        -delta[0] * sin_y - delta[1] * cos_y,
        -delta[2],
    ])
    t_cam_prev = -R.T @ t_arr
    exp_norm = float(np.linalg.norm(expected))
    t_cp_norm = float(np.linalg.norm(t_cam_prev))
    denom = max(t_cp_norm * exp_norm, 1e-9)
    cos_angle = float(np.dot(t_cam_prev, expected) / denom)
    return {
        'i': int(i),
        'status': 'ok',
        't_a': float(t_arr[0]),
        't_b': float(t_arr[1]),
        't_z': float(t_arr[2]),
        't_xy_norm': float(np.hypot(t_arr[0], t_arr[1])),
        'gps_disp_m': float(displacements[i]),
        'expected_xy_norm_m': float(np.hypot(expected[0], expected[1])),
        'cos_angle_t_vs_expected': cos_angle,
        'flipped': int(flags["flipped"]),
        'planar_fallback': int(flags["planar_fallback"]),
    }


def _write_vo_diagnostics(rows, cfg: Config):
    """Salva tutti i diag in un CSV in output/. Una riga per frame i ≥ 1."""
    if not rows:
        return
    out_dir = cfg.resolve(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"vo_diag_{cfg.start}_to_{cfg.end}.csv"
    fieldnames = [
        'i', 'status', 't_a', 't_b', 't_z', 't_xy_norm',
        'gps_disp_m', 'expected_xy_norm_m', 'cos_angle_t_vs_expected',
        'flipped', 'planar_fallback',
    ]
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f"VO diag: salvati {len(rows)} record in {path}")


def _print_vo_summary(rows):
    """Stampa una sintesi: ‖t_xy‖ mediana, cos angolo mediano, |t_z| mediano."""
    ok = [r for r in rows if r.get('status') == 'ok']
    if not ok:
        print("VO summary: nessun frame con posa OK.")
        return
    t_xy = np.array([r['t_xy_norm'] for r in ok])
    t_z = np.array([abs(r['t_z']) for r in ok])
    cos_a = np.array([r['cos_angle_t_vs_expected'] for r in ok])
    print(
        f"VO summary ({len(ok)} frame OK): "
        f"||t_xy|| mediana={np.median(t_xy):.3f} (1.0 = tutto in piano), "
        f"|t_z| mediano={np.median(t_z):.3f}, "
        f"cos(t,expected) mediano={np.median(cos_a):.3f} (1.0 = direzione perfetta)."
    )


def _expected_R_from_yaw(yaw_prev_deg: float, yaw_curr_deg: float) -> np.ndarray:
    """Rotazione cam_prev → cam_new attesa dal solo yaw del metadato (nadir + FollowYaw).

    Per gimbal nadir solidale al body via FollowYaw, fra due frame consecutivi la
    rotazione fra le due frame camera è puramente attorno a z_cam (image-forward,
    cioè l'asse verticale a terra). Con la convenzione assi
    `x_cam = compass(yaw+90°), y_cam = compass(yaw+180°)` usata in tutto il
    progetto, la R relativa è:

        R = [[ cos Δ, sin Δ, 0],
             [-sin Δ, cos Δ, 0],
             [ 0,     0,     1]]    con Δ = yaw_curr - yaw_prev (gradi cw).

    Si usa come prior per `decomposeEssentialMat` quando `recoverPose` cade
    nella degenerazione planare e seleziona la rotazione sbagliata.
    """
    delta = np.radians(yaw_curr_deg - yaw_prev_deg)
    c, s = np.cos(delta), np.sin(delta)
    return np.array([
        [c, s, 0.0],
        [-s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _disambiguate_t_sign(R, t, geometry, i, *, t_xy_min: float = 0.5):
    """Allinea `t` (output di `decomposeEssentialMat`) alla direzione GPS attesa.

    `decomposeEssentialMat` produce (R, t) tali che x_cam_new = R·x_cam_prev + t.
    La traslazione del centro camera espressa nel frame cam_prev è -R^T·t. La
    confrontiamo con il delta GPS atteso, ruotato dal mondo ENU al frame
    cam_prev usando lo yaw del frame i-1 (gimbal nadir + FollowYaw).

    Convenzione image axes (consistente con `_vo_to_enu`):
        x_cam = image-right  →  world  ( cos yaw, -sin yaw)
        y_cam = image-down   →  world  (-sin yaw, -cos yaw)
        z_cam → -Up

    Due casi patologici da gestire prima del confronto del segno:
      1. Drone quasi fermo (‖delta_xy‖ < 0.2 m): nessun segnale GPS utile,
         si accetta `t` invariato.
      2. Scena planare nadir degenere (‖t_xy‖ < `t_xy_min`): l'essential matrix
         su volo orizzontale + nadir ammette soluzioni con tutto il modulo
         unitario di `t` lungo l'asse ottico (z_cam = verticale a terra). Una
         volta scalato per ‖delta_GPS‖ questo concentra lo spostamento metrico
         in altitudine invece che nel piano, accorciando la traiettoria
         orizzontale. In quei frame sostituiamo `t` con la direzione GPS+yaw
         attesa: `t = -R · expected_unit` (deriva da t_cam_prev = -R^T·t).

    Ritorna `(t, flags)` con `flags = {"flipped": bool, "planar_fallback": bool}`.
    """
    flags = {"flipped": False, "planar_fallback": False}
    delta = geometry.positions_enu[i] - geometry.positions_enu[i - 1]
    if float(np.linalg.norm(delta[:2])) < 0.2:
        return t, flags
    yaw_prev = float(np.radians(geometry.yaw_deg[i - 1]))
    cos_y, sin_y = np.cos(yaw_prev), np.sin(yaw_prev)
    expected = np.array([
         delta[0] * cos_y - delta[1] * sin_y,
        -delta[0] * sin_y - delta[1] * cos_y,
        -delta[2],
    ])
    t_arr = np.asarray(t).ravel()
    if float(np.hypot(t_arr[0], t_arr[1])) < t_xy_min:
        expected_unit = expected / max(float(np.linalg.norm(expected)), 1e-9)
        t_new = (-R @ expected_unit).reshape(np.asarray(t).shape)
        flags["planar_fallback"] = True
        return t_new, flags
    t_cam_prev = -R.T @ t_arr
    if float(np.dot(t_cam_prev, expected)) < 0:
        flags["flipped"] = True
        return -t, flags
    return t, flags
