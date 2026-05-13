"""Visual odometry: stima R, t dalla matrice essenziale e accumulo della traiettoria."""
import cv2
import numpy as np

from .timing import timed


@timed
def estimate_pose(q1, q2, camera_matrix, expected_R=None, ransac_thr=0.5, prob=0.999):
    """Stima rotazione e traslazione tra due viste dai matching q1→q2.

    Ritorna (R, t) come (3x3, 3x1) oppure (None, None) se non determinabile.
    `t` è normalizzato a norma unitaria; il chiamante lo moltiplica per la scala
    metrica desiderata e ne disambigua il segno (vedi `_disambiguate_t_sign`).

    Su scena planare nadir l'essential matrix è degenere e `cv2.recoverPose` può
    selezionare una delle due decomposizioni mathematicamente valide ma fisicamente
    sbagliata, tipicamente con R ≈ rotazione di 180° attorno a z_cam (image-forward).
    Una volta accumulata in `Trajectory.cur_pose` questa rotazione spuria specchia
    la traiettoria di 180° per *tutti* i frame successivi (impossibile recuperare).
    Quando `expected_R` è fornita (calcolata dalla differenza di yaw del metadato),
    si bypassa `recoverPose` e si sceglie tra le due R candidate di
    `decomposeEssentialMat` quella più vicina in norma di Frobenius. Il segno di
    `t` resta ambiguo qui e viene risolto a valle.
    """
    if q1 is None or q2 is None or len(q1) < 5 or len(q2) < 5:
        return None, None

    E, _ = cv2.findEssentialMat(q1, q2, camera_matrix,
                                method=cv2.RANSAC, prob=prob, threshold=ransac_thr)
    if E is None:
        return None, None

    if expected_R is None:
        _, R, t, _ = cv2.recoverPose(E, q1, q2, camera_matrix)
        return R, t

    R1, R2, t = cv2.decomposeEssentialMat(E)
    err1 = np.linalg.norm(R1 - expected_R, ord='fro')
    err2 = np.linalg.norm(R2 - expected_R, ord='fro')
    R = R1 if err1 < err2 else R2
    return R, t


class Trajectory:
    """Posa cumulativa integrando trasformazioni relative consecutive.

    Espone `path` (lista di (x, y, z)).
    """

    def __init__(self):
        self.cur_pose = np.eye(4)
        self.path = [(0.0, 0.0, 0.0)]

    @timed
    def update(self, R_rel, t_rel):
        """Concatena la trasformazione relativa (R, t) alla posa corrente.

        `cv2.recoverPose` restituisce (R, t) tali che x_cam2 = R·x_cam1 + t
        (cioè la trasformazione *da* cam_prev *a* cam_new). Per accumulare
        `cur_pose` nella convenzione cam-in-world va usata la posa inversa,
        ossia la posa di cam_new espressa nel frame di cam_prev:
            inv([R | t]) = [Rᵀ | -Rᵀ·t]
        Senza l'inversione la traiettoria viene specchiata di 180°.
        """
        t_rel = np.asarray(t_rel).ravel()
        T_inv = np.eye(4)
        T_inv[:3, :3] = R_rel.T
        T_inv[:3, 3] = -R_rel.T @ t_rel
        self.cur_pose = self.cur_pose @ T_inv

        x, y, z = self.cur_pose[:3, 3]
        self.path.append((float(x), float(y), float(z)))
