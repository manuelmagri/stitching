"""Visual odometry: stima R, t dalla matrice essenziale e accumulo della traiettoria."""
import cv2
import numpy as np

from .timing import timed


@timed
def estimate_pose(q1, q2, camera_matrix, ransac_thr=0.5, prob=0.999):
    """Stima rotazione e traslazione tra due viste dai matching q1→q2.

    Ritorna (R, t) come (3x3, 3x1) oppure (None, None) se non determinabile.
    `t` è normalizzato a norma unitaria; il chiamante lo moltiplica per la scala
    metrica desiderata (tipicamente `scales[i]`).
    """
    if q1 is None or q2 is None or len(q1) < 5 or len(q2) < 5:
        return None, None

    E, _ = cv2.findEssentialMat(q1, q2, camera_matrix,
                                method=cv2.RANSAC, prob=prob, threshold=ransac_thr)
    if E is None:
        return None, None

    _, R, t, _ = cv2.recoverPose(E, q1, q2, camera_matrix)
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
