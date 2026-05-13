"""Orchestratore della pipeline: stitching + visual odometry.

Lo stitching ha due backend selezionabili via `cfg.mosaic_method`:
  - "ortho" (default): ortomosaico geo-riferito da GPS + gimbal yaw, con
    AGL auto-calibrata. Drift zero per costruzione, output metrico.
  - "scans": `cv2.Stitcher_SCANS`, pipeline solo-pixel di OpenCV. Più
    permissiva ma soggetta a drift su sequenze lunghe.

La VO è indipendente: estrae feature ORB su griglia, matcha pairwise e
stima R, t con la matrice essenziale.
"""
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
from .ortho import render_ortho
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

    mosaic = _render_mosaic(images, K, geometry, cfg)

    mosaic_path = out_dir / f"mosaic_{cfg.start}_to_{cfg.end}.jpg"
    cv2.imwrite(str(mosaic_path), mosaic)
    print(f"Mosaico salvato in {mosaic_path}")

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
        vo_path_enu = _vo_to_enu(trajectory.path, cfg)
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
    """Compone il mosaico col backend selezionato da cfg.mosaic_method."""
    method = cfg.mosaic_method
    if method == "scans":
        print(f"Stitching di {len(images)} immagini con cv2.Stitcher SCANS...")
        return render_mosaic(images, confidence_thresh=cfg.stitcher_confidence_thresh)

    if method == "ortho":
        print(f"Ortomosaico geo-riferito di {len(images)} immagini...")
        result = render_ortho(
            images, geometry, K,
            agl=cfg.ortho_agl_m,
            gsd=cfg.ortho_gsd,
        )
        return result.image

    raise ValueError(f"cfg.mosaic_method sconosciuto: {method!r}")


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
            continue

        R, t = estimate_pose(
            q_prev, q_curr, K,
            ransac_thr=cfg.essential_ransac_thr,
            prob=cfg.essential_ransac_prob,
        )
        if R is not None:
            t, flipped = _disambiguate_t_sign(R, t, geometry, i)
            if flipped:
                n_flips += 1
            trajectory.update(R, t * float(displacements[i]))
        else:
            print(f"VO: posa non stimabile al frame {i}, traiettoria invariata.")

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
    return trajectory


def _disambiguate_t_sign(R, t, geometry, i):
    """Allinea il segno di `t` (output di `recoverPose`) alla direzione GPS attesa.

    `recoverPose` produce (R, t) tali che x_cam_new = R·x_cam_prev + t. La
    traslazione del centro camera espressa nel frame cam_prev è -R^T·t. La
    confrontiamo con il delta GPS atteso, ruotato dal mondo ENU al frame
    cam_prev usando lo yaw del frame i-1 (gimbal nadir + FollowYaw).

    Convenzione image axes (consistente con `_vo_to_enu`):
        x_cam = image-right  →  world  ( cos yaw, -sin yaw)
        y_cam = image-down   →  world  (-sin yaw, -cos yaw)
        z_cam → -Up

    Ritorna (t, flipped) dove `flipped` è True se è stato invertito. Quando il
    drone è quasi fermo non c'è segnale GPS utile e si accetta `t` invariato.
    """
    delta = geometry.positions_enu[i] - geometry.positions_enu[i - 1]
    if float(np.linalg.norm(delta[:2])) < 0.2:
        return t, False
    yaw_prev = float(np.radians(geometry.yaw_deg[i - 1]))
    cos_y, sin_y = np.cos(yaw_prev), np.sin(yaw_prev)
    expected = np.array([
         delta[0] * cos_y - delta[1] * sin_y,
        -delta[0] * sin_y - delta[1] * cos_y,
        -delta[2],
    ])
    t_cam_prev = -R.T @ np.asarray(t).ravel()
    if float(np.dot(t_cam_prev, expected)) < 0:
        return -t, True
    return t, False
