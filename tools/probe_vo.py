"""Probe diagnostico per i file in data/: scopre cosa rappresentano rotations.txt,
translations.txt, yaws.txt e scales.txt confrontando le traiettorie ricostruite col GPS.

Per ogni interpretazione plausibile delle traslazioni:
1. Ricostruisce la traiettoria 2D
2. La allinea al GPS via Procrustes 2D (rotazione + traslazione + scala globale)
3. Stampa RMSE in metri

L'interpretazione con RMSE piu' basso e' quella giusta.

Stampa anche differenze fra yaws.txt e flight_yaw EXIF.
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import io_data  # noqa: E402
from utils.geodesy import make_transformers  # noqa: E402


def parse_translations(path: Path) -> np.ndarray:
    out: list[list[float]] = []
    pat = re.compile(r"\[([^\]]*)\]")
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if not m:
            continue
        vals = [float(v) for v in m.group(1).split() if v]
        if len(vals) == 3:
            out.append(vals)
    return np.array(out, dtype=np.float64)


def parse_rotations(path: Path) -> np.ndarray:
    text = path.read_text()
    blocks = re.split(r"Rotation Matrix \d+:", text)[1:]
    out: list[list[list[float]]] = []
    for blk in blocks:
        rows: list[list[float]] = []
        for raw in blk.strip().splitlines():
            line = raw.strip().lstrip("[").rstrip("]").strip()
            if not line:
                continue
            vals = [float(v) for v in line.split() if v not in ("", "[", "]")]
            if len(vals) == 3:
                rows.append(vals)
            if len(rows) == 3:
                break
        if len(rows) == 3:
            out.append(rows)
    return np.array(out, dtype=np.float64)


def parse_scalar_lines(path: Path) -> np.ndarray:
    out: list[float] = []
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        rhs = line.split(":", 1)[1].strip()
        if rhs.startswith("["):
            continue
        try:
            out.append(float(rhs))
        except ValueError:
            pass
    return np.array(out, dtype=np.float64)


def procrustes_2d(A: np.ndarray, B: np.ndarray) -> float:
    """RMSE dopo fit rigid+scale globale: s*R@A + t -> B."""
    cA = A.mean(axis=0)
    cB = B.mean(axis=0)
    Ac = A - cA
    Bc = B - cB
    H = Ac.T @ Bc
    U, S_vals, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    var_A = (Ac ** 2).sum()
    scale = S_vals.sum() / var_A if var_A > 1e-12 else 1.0
    B_pred = (scale * (Ac @ R.T)) + cB
    return float(np.sqrt(((B_pred - B) ** 2).sum(axis=1).mean()))


def rot_z_2d(deg: float) -> np.ndarray:
    th = np.deg2rad(deg)
    c, s = np.cos(th), np.sin(th)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def main() -> None:
    data_dir = ROOT / "data"

    records = io_data.load_metadata(data_dir / "metadati.txt")
    n = len(records)
    lat0 = sum(r["lat"] for r in records) / n
    lon0 = sum(r["lon"] for r in records) / n
    to_utm, _, _ = make_transformers(lat0, lon0)
    positions_gps = np.array(
        [to_utm.transform(r["lon"], r["lat"]) for r in records]
    )
    gps_xy = positions_gps - positions_gps[0]

    trans = parse_translations(data_dir / "translations.txt")
    rots = parse_rotations(data_dir / "rotations.txt")
    yaws = parse_scalar_lines(data_dir / "yaws.txt")
    scales = parse_scalar_lines(data_dir / "scales.txt")

    print(f"[probe] frame in metadati: {n}")
    print(f"[probe] translations: {trans.shape}")
    print(f"[probe] rotations: {rots.shape}")
    print(f"[probe] yaws: {yaws.shape}")
    print(f"[probe] scales: {scales.shape}")
    print()

    flight_yaws = np.array([r["flight_yaw_deg"] for r in records])
    yaw_diff = (yaws - flight_yaws + 180.0) % 360.0 - 180.0
    print("[probe] Confronto yaws.txt vs flight_yaw EXIF:")
    print(
        f"  mean={yaw_diff.mean():+.2f} deg  median={np.median(yaw_diff):+.2f} deg  "
        f"std={yaw_diff.std():.2f} deg  max|d|={np.abs(yaw_diff).max():.2f} deg"
    )
    print()

    if not (n == len(trans) == len(rots) == len(yaws) == len(scales)):
        print(f"[probe] ATTENZIONE: lunghezze non coincidono.")
        m = min(n, len(trans), len(rots), len(yaws), len(scales))
        gps_xy = gps_xy[:m]
        trans = trans[:m]
        rots = rots[:m]
        yaws = yaws[:m]
        scales = scales[:m]

    nuse = len(gps_xy)

    traj_A = trans[:, :2].copy()

    traj_B = np.cumsum(trans[:, :2], axis=0)

    traj_C = np.zeros((nuse, 2))
    pos3 = np.zeros(3)
    for i in range(nuse):
        if i == 0:
            pos3 = trans[i].copy()
        else:
            pos3 = pos3 + rots[i - 1] @ trans[i]
        traj_C[i] = pos3[:2]

    traj_D = np.zeros((nuse, 2))
    pos3 = np.zeros(3)
    for i in range(nuse):
        pos3 = pos3 + rots[i] @ trans[i]
        traj_D[i] = pos3[:2]

    traj_E = np.zeros((nuse, 2))
    pos3 = np.zeros(3)
    for i in range(nuse):
        if i == 0:
            pos3 = trans[i] * scales[i]
        else:
            pos3 = pos3 + rots[i - 1] @ (trans[i] * scales[i])
        traj_E[i] = pos3[:2]

    traj_F = np.zeros((nuse, 2))
    pos2 = np.zeros(2)
    for i in range(nuse):
        Ry = rot_z_2d(yaws[i - 1] if i > 0 else yaws[0])
        if i == 0:
            pos2 = trans[i, :2].copy()
        else:
            pos2 = pos2 + Ry @ trans[i, :2]
        traj_F[i] = pos2

    traj_G = np.zeros((nuse, 2))
    pos2 = np.zeros(2)
    for i in range(nuse):
        Ry = rot_z_2d(yaws[i])
        pos2 = pos2 + Ry @ trans[i, :2]
        traj_G[i] = pos2

    traj_H = np.zeros((nuse, 2))
    pos3 = np.zeros(3)
    for i in range(nuse):
        if i == 0:
            pos3 = trans[i].copy()
        else:
            pos3 = pos3 + rots[i - 1].T @ trans[i]
        traj_H[i] = pos3[:2]

    hypotheses = {
        "A: trans[:,:2] gia' assoluto": traj_A,
        "B: cumsum trans[:,:2] (world frame)": traj_B,
        "C: cumsum R[i-1] @ trans[i] (prev-cam frame)": traj_C,
        "D: cumsum R[i] @ trans[i] (curr-cam frame)": traj_D,
        "E: cumsum R[i-1] @ (trans[i]*scale[i])": traj_E,
        "F: cumsum R_yaw2D(prev) @ trans[i,:2]": traj_F,
        "G: cumsum R_yaw2D(curr) @ trans[i,:2]": traj_G,
        "H: cumsum R[i-1].T @ trans[i] (inverso)": traj_H,
    }

    print(f"[probe] Lunghezza traiettoria GPS: {np.linalg.norm(gps_xy[-1]):.1f} m "
          f"(end-to-start), totale path {np.sum(np.linalg.norm(np.diff(gps_xy, axis=0), axis=1)):.1f} m")
    print()
    print(f"{'Hypothesis':<55} {'RMSE_align':>11}  {'Length':>9}")
    print("-" * 80)
    for label, traj in hypotheses.items():
        rmse = procrustes_2d(traj, gps_xy)
        path_len = float(np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)))
        print(f"{label:<55} {rmse:>10.2f}m  {path_len:>8.1f}m")
    print()
    print("L'ipotesi col RMSE piu' basso e' quella corretta.")
    print("La 'Length' (path totale) confrontata con quella GPS dice se servono gli scales.")


if __name__ == "__main__":
    main()
