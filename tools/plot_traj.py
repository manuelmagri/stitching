"""Genera l'HTML interattivo con la traiettoria GPS sovrapposta alla VO integrata.

Lancia: `python tools/plot_traj.py` -> apre output/trajectory.html

Origine = primo frame del dataset (entrambe le tracce). Cosi' le differenze
visibili sono drift e bias puri, non offset di sistema di riferimento.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import flight_filter, io_data, odometry  # noqa: E402
from utils.geodesy import make_transformers  # noqa: E402
from visualization.trajectory_plot import plot_trajectories  # noqa: E402


def main() -> None:
    data_dir = ROOT / "data"
    out_path = ROOT / "output" / "trajectory.html"

    records = io_data.load_metadata(data_dir / "metadati.txt")
    flight_filter.mark_curves(records, roll_threshold_deg=10.0, yaw_rate_threshold_deg=5.0)

    vo = odometry.load_vo_deltas(data_dir / "translations.txt")
    if len(vo) != len(records):
        raise RuntimeError(
            f"VO/metadata mismatch: {len(vo)} delta vs {len(records)} record"
        )

    lat0 = sum(r["lat"] for r in records) / len(records)
    lon0 = sum(r["lon"] for r in records) / len(records)
    to_utm, _, _ = make_transformers(lat0, lon0)
    gps_xy = np.array([to_utm.transform(r["lon"], r["lat"]) for r in records])
    gps_xy = gps_xy - gps_xy[0]  # origine = primo frame

    # VO cumulativa: vo gia' in (east, north) dopo il fix in load_vo_deltas
    vo_xy = np.cumsum(vo, axis=0)

    curve_idx = [i for i, r in enumerate(records) if r["is_curve"]]

    n_gps = np.linalg.norm(gps_xy[-1])
    path_gps = float(np.sum(np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)))
    path_vo = float(np.sum(np.linalg.norm(np.diff(vo_xy, axis=0), axis=1)))
    print(f"[plot] frame: {len(records)}, curve: {len(curve_idx)}")
    print(f"[plot] path totale GPS: {path_gps:.1f} m | VO: {path_vo:.1f} m")
    print(f"[plot] distanza end<->start GPS: {n_gps:.1f} m")
    print(f"[plot] drift finale VO vs GPS: "
          f"{np.linalg.norm(gps_xy[-1] - vo_xy[-1]):.2f} m")

    plot_trajectories(
        gps_points=gps_xy,
        vo_points=vo_xy,
        output_path=str(out_path),
        title="Traiettoria GPS vs VO (entrambe in UTM locale, origine = frame 1)",
        curve_indices=curve_idx,
    )
    print(f"[plot] HTML: {out_path}")


if __name__ == "__main__":
    main()
