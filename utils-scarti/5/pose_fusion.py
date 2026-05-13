"""Fusione delle pose: GPS UTM (posizione) + IMU/gimbal (orientazione).

Filosofia di questa pass:

  - **Posizione** del centro ottico = UTM-relativo del frame (gia' calcolato
    da `gps_utm.proietta_frames`). GPS L1 da' precisione metrica e non
    accumula drift: per un mosaico aereo coerente in UTM e' la scelta
    piu' robusta. Salto la fusione Kalman complessa.
  - **Orientazione** = angoli IMU della camera (gimbal di default, drone
    body come fallback). DJI XMP fornisce yaw/pitch/roll gia' nel
    riferimento mondo (yaw: 0=N, +CW; pitch nadir ~ -90 deg).
  - **VO** entra solo come *quality flag*: se gli inlier sono pochi o se
    |t_VO| differisce molto da |Delta UTM|, il frame viene marcato
    `confident=False`. Il caller (mosaic) puo' decidere se saltarlo o
    pesarlo meno nel blending.

In una pass successiva, qui si puo' inserire un filtro di Kalman o un
bundle adjustment: l'interfaccia `WorldPose` resta la stessa.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .gps_utm import UtmFrame
from .io_loader import FrameMeta
from .vo import VoStep


YawSource = Literal["gimbal", "flight"]


@dataclass(frozen=True)
class WorldPose:
    """Posa del centro camera nel riferimento mondo UTM-locale.

    Le coordinate `east_m`, `north_m` sono RELATIVE all'origine del
    proiettore UTM (cioe' coerenti con `UtmFrame.east_rel/north_rel`).
    """

    index: int                  # indice globale del frame (= FrameMeta.index)
    east_m: float
    north_m: float
    alt_m: float

    yaw_deg: float              # 0 = Nord, + = senso orario (DJI)
    pitch_deg: float            # nadir = -90
    roll_deg: float

    R_world_cam: np.ndarray     # 3x3 float64, cam -> mondo (ENU)
    confident: bool             # quality flag (vedi modulo docstring)


# --------------------------------------------------------------------------- #
# Rotazioni IMU -> matrice
# --------------------------------------------------------------------------- #

def _Rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _Ry(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _Rx(deg: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _R_world_cam(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Costruisce R che mappa il frame camera (x:dx, y:giu', z:avanti) in ENU.

    Convenzione DJI:
      - yaw 0 = Nord, +CW (cioe' verso est).
      - pitch 0 = camera orizzontale, -90 = nadir (verso il basso).
      - roll 0 = orizzonte allineato.

    Ordine euleriano applicato: yaw (asse Up) -> pitch (asse East corpo)
    -> roll (asse Nord corpo). Per yaw=0, pitch=-90, roll=0 (caso tipico)
    la matrice manda:
       cam_x (destra) -> +East
       cam_y (giu')   -> -North (verso sud, perche' la camera guarda giu')
       cam_z (avanti) -> -Up (verso il suolo)
    """
    # NB: nel sistema DJI yaw CW visto da sopra equivale a una rotazione
    # NEGATIVA attorno all'asse Up (asse Z di ENU). Da qui il segno meno.
    return _Rz(-yaw_deg) @ _Ry(pitch_deg) @ _Rx(roll_deg)


# --------------------------------------------------------------------------- #
# Fusione
# --------------------------------------------------------------------------- #

def _scale_check(vo_t: np.ndarray, gps_step_m: float,
                 tol_rel: float = 0.5) -> bool:
    """True se |t_VO| e' coerente con |Delta UTM| (entro `tol_rel`)."""
    vo_norm = float(np.linalg.norm(vo_t))
    if gps_step_m < 1e-3:
        # Drone fermo: niente da confrontare; la rotazione vale comunque.
        return vo_norm < 0.5
    return abs(vo_norm - gps_step_m) <= tol_rel * gps_step_m


def fuse(frames: list[FrameMeta],
         utm_frames: list[UtmFrame],
         vo_steps: list[VoStep] | None = None,
         yaw_source: YawSource = "gimbal",
         min_vo_inliers: int = 30) -> list[WorldPose]:
    """Produce una lista di `WorldPose`, una per frame.

    Parameters
    ----------
    frames, utm_frames
        Devono avere la stessa lunghezza ed essere allineati (i-esimo
        elemento riferito allo stesso scatto).
    vo_steps
        Lunghezza len(frames)-1 (uno step per ogni coppia consecutiva).
        Se None, ogni frame e' marcato `confident=True`.
    yaw_source
        "gimbal" (default, raccomandato per camera stabilizzata) oppure
        "flight" (yaw del drone, utile se il gimbal e' assente o bloccato).
    min_vo_inliers
        Sotto questa soglia il frame i viene marcato `confident=False`.
    """
    if len(frames) != len(utm_frames):
        raise ValueError(
            f"frames ({len(frames)}) e utm_frames ({len(utm_frames)}) "
            "devono avere la stessa lunghezza"
        )
    if vo_steps is not None and len(vo_steps) != len(frames) - 1:
        raise ValueError(
            f"vo_steps deve avere lunghezza {len(frames)-1}, ricevuto {len(vo_steps)}"
        )

    poses: list[WorldPose] = []
    for i, (fm, uf) in enumerate(zip(frames, utm_frames)):
        if yaw_source == "gimbal":
            yaw = fm.gimbal_yaw_deg
            pitch = fm.gimbal_pitch_deg
            roll = fm.gimbal_roll_deg
        else:
            yaw = fm.flight_yaw_deg
            pitch = fm.flight_pitch_deg
            roll = fm.flight_roll_deg

        # Quality flag: combinazione di inlier VO e coerenza di scala.
        confident = True
        if vo_steps is not None and i > 0:
            step = vo_steps[i - 1]
            if not step.ok or step.n_inliers < min_vo_inliers:
                confident = False
            else:
                gps_step = float(np.hypot(
                    uf.east_rel - utm_frames[i - 1].east_rel,
                    uf.north_rel - utm_frames[i - 1].north_rel,
                ))
                if not _scale_check(step.t, gps_step):
                    confident = False

        poses.append(WorldPose(
            index=fm.index,
            east_m=uf.east_rel,
            north_m=uf.north_rel,
            alt_m=uf.alt_m,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            R_world_cam=_R_world_cam(yaw, pitch, roll),
            confident=confident,
        ))

    return poses


def yaw_unwrap(poses: list[WorldPose]) -> np.ndarray:
    """Yaw "unwrapped" (continuo, senza salti +/-180) per analisi/grafici.

    Non modifica le pose: ritorna solo l'array (gradi). Utile in
    visualization quando si plotta la traiettoria angolare.
    """
    y = np.array([p.yaw_deg for p in poses], dtype=np.float64)
    return np.degrees(np.unwrap(np.deg2rad(y)))
