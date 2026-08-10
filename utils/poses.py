"""Pose degli scatti sul canvas: seed dai sensori, raffinamento globale dalle immagini.

Ogni posa e' una similarita' 2D (tx, ty, theta, log_scala) che porta i pixel dell'immagine
sui pixel del canvas. Il seed viene dal frame locale di `utils.localframe`; il
raffinamento e' un minimi quadrati sparso che concilia seed, odometria e vincoli
fotografici.

I residui sono TUTTI in pixel di canvas, anche quelli angolari e di scala. Un errore di
rotazione `dtheta` sposta di `R*dtheta` un punto a distanza R dal centro, e un errore
relativo di scala `dlog_s` lo sposta di `R*dlog_s`: moltiplicando per il semidiagonale
dell'immagine si ottengono grandezze confrontabili, e i pesi restano numeri puri. Senza
questa conversione un'ancora angolare pesata 1 varrebbe 0,017 per grado, cioe' niente
rispetto a residui fotografici di qualche pixel, e non fisserebbe un bel nulla.

Chi fa cosa, senza GPS:

    ancora su theta   <- bussola. Fissa la rotazione assoluta, quindi il nord.
    ancora su log_s   <- quota barometrica e focale. Fissa la scala assoluta.
    ancora su tx, ty  <- odometria integrata. Debole: fissa i due gradi di liberta' della
                         traslazione globale e impedisce a un frame senza vincoli di
                         andare alla deriva. Non e' una misura di cui fidarsi.
    vincoli VO        <- odometria fra scatti consecutivi. E' cio' che attraversa le
                         virate, dove non ci sono coppie fotografiche.
    vincoli fotografici <- le immagini. Sono i piu' precisi e devono dominare.
"""
import math

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


def initial_pose(
    position_m: tuple[float, float],
    yaw_deg: float,
    gsd_frame: float,
    gsd_canvas: float,
    image_size: tuple[int, int],
    origin_m: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Similarita' 3x3: pixel immagine -> pixel canvas.

    Il canvas ha x verso est e y verso SUD, come tutte le immagini raster, quindi il nord
    entra col segno meno. `gsd_frame / gsd_canvas` compensa le variazioni di quota fra uno
    scatto e l'altro.
    """
    est, nord = position_m
    cx = (est - origin_m[0]) / gsd_canvas
    cy = -(nord - origin_m[1]) / gsd_canvas

    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    scala = gsd_frame / gsd_canvas
    w, h = image_size

    al_centro = np.array([[1.0, 0.0, -w / 2.0], [0.0, 1.0, -h / 2.0], [0.0, 0.0, 1.0]])
    ruota = np.array(
        [[scala * c, -scala * s, 0.0], [scala * s, scala * c, 0.0], [0.0, 0.0, 1.0]]
    )
    trasla = np.array([[1.0, 0.0, cx], [0.0, 1.0, cy], [0.0, 0.0, 1.0]])
    return trasla @ ruota @ al_centro


def initial_poses(
    positions_m: np.ndarray,
    records: list[dict],
    gsds: np.ndarray,
    gsd_canvas: float,
    image_size: tuple[int, int],
) -> list[np.ndarray]:
    """Pose iniziali di tutti i record, con origine sul primo."""
    return [
        initial_pose(
            tuple(positions_m[i]),
            records[i]["flight_yaw_deg"],
            float(gsds[i]),
            gsd_canvas,
            image_size,
        )
        for i in range(len(records))
    ]


def rescale_poses(transforms: list[np.ndarray], factor: float) -> list[np.ndarray]:
    """Porta le pose da una risoluzione all'altra moltiplicando `factor` volte.

    Se immagine e canvas cambiano risoluzione dello stesso fattore s, un punto p diventa
    s*p in entrambi, quindi la posa diventa S M S^-1 con S = diag(s, s, 1). Per una
    similarita' questo lascia intatte rotazione e scala e moltiplica la sola traslazione.
    E' l'unico ponte fra la passata a risoluzione ridotta, che stima le pose, e quella a
    piena risoluzione, che compone il mosaico.
    """
    riscalate = []
    for M in transforms:
        N = M.copy()
        N[0, 2] *= factor
        N[1, 2] *= factor
        riscalate.append(N)
    return riscalate


def _decompose(M: np.ndarray) -> tuple[float, float, float, float]:
    a, b = float(M[0, 0]), float(M[1, 0])
    scala = math.hypot(a, b)
    return float(M[0, 2]), float(M[1, 2]), math.atan2(b, a), math.log(max(scala, 1e-12))


def _compose(tx: float, ty: float, theta: float, log_s: float) -> np.ndarray:
    s = math.exp(log_s)
    c, sn = math.cos(theta), math.sin(theta)
    return np.array(
        [[s * c, -s * sn, tx], [s * sn, s * c, ty], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def _build_matrices(params: np.ndarray, n: int) -> np.ndarray:
    p = params.reshape(n, 4)
    s = np.exp(p[:, 3])
    c, sn = np.cos(p[:, 2]), np.sin(p[:, 2])
    Ms = np.zeros((n, 3, 3), dtype=np.float64)
    Ms[:, 0, 0] = s * c
    Ms[:, 0, 1] = -s * sn
    Ms[:, 1, 0] = s * sn
    Ms[:, 1, 1] = s * c
    Ms[:, 0, 2] = p[:, 0]
    Ms[:, 1, 2] = p[:, 1]
    Ms[:, 2, 2] = 1.0
    return Ms


def refine_poses(
    initial_M: list[np.ndarray],
    pair_constraints: list[tuple[int, int, np.ndarray, int]],
    image_size: tuple[int, int],
    vo_constraints: list[tuple[int, int, float, float]] | None = None,
    vo_weight: float = 0.5,
    yaw_weight: float = 0.5,
    scale_weight: float = 0.5,
    seed_weight: float = 0.02,
    max_iter: int = 40,
    verbose: int = 1,
) -> list[np.ndarray]:
    """Pose raffinate, stessa lunghezza e ordine di `initial_M`.

    `pair_constraints` sono (i, j, H_ij, n_inlier) con H_ij che mappa i pixel dell'immagine
    i su quelli della j; il peso e' la radice del numero di inlier, cosi' una coppia con
    molte corrispondenze conta piu' di una marginale.

    `vo_constraints` sono (i, j, dx, dy): lo spostamento atteso del centro immagine fra i
    e j, in pixel di canvas.

    Sul peso dell'odometria: il valore storico 5.0 era tarato su una odometria visiva che
    si credeva precisa al decimetro. Quella effettivamente disponibile sbaglia 0,46 m per
    passo, cioe' una dozzina di pixel alla risoluzione di lavoro, contro l'uno o due pixel
    di un vincolo fotografico. Da qui il valore molto piu' basso: l'odometria serve dove le
    immagini tacciono, non a correggerle.
    """
    n = len(initial_M)
    if n == 0:
        return []

    w, h = image_size
    vo_pairs = list(vo_constraints) if vo_constraints else []
    raggio = 0.5 * math.hypot(w, h)  # ponte fra residui angolari e residui in pixel

    corners = np.array(
        [[0.0, 0.0, 1.0], [w, 0.0, 1.0], [w, h, 1.0], [0.0, h, 1.0]], dtype=np.float64
    )
    centro = np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float64)

    x0 = np.zeros(n * 4, dtype=np.float64)
    for i, M in enumerate(initial_M):
        x0[i * 4 : i * 4 + 4] = _decompose(M)
    seed = x0.reshape(n, 4).copy()

    # Vincoli fotografici, pre-elaborati: gli angoli dell'immagine i visti nel sistema
    # della j, cosi' il residuo e' una sola moltiplicazione per posa.
    if pair_constraints:
        idx_i = np.array([c[0] for c in pair_constraints], dtype=int)
        idx_j = np.array([c[1] for c in pair_constraints], dtype=int)
        pesi_coppie = np.sqrt(
            np.maximum([c[3] for c in pair_constraints], 1)
        ).astype(np.float64)[:, None, None]
        angoli_in_j = np.empty((len(pair_constraints), 4, 3), dtype=np.float64)
        for k, (_, _, H, _) in enumerate(pair_constraints):
            proiettati = (corners @ np.asarray(H, dtype=np.float64).T)[:, :2]
            angoli_in_j[k] = np.column_stack([proiettati, np.ones(4)])
    else:
        idx_i = idx_j = np.empty(0, dtype=int)
        pesi_coppie = np.empty((0, 1, 1))
        angoli_in_j = np.empty((0, 4, 3))

    if vo_pairs:
        vo_i = np.array([c[0] for c in vo_pairs], dtype=int)
        vo_j = np.array([c[1] for c in vo_pairs], dtype=int)
        vo_delta = np.array([[c[2], c[3]] for c in vo_pairs], dtype=np.float64)
    else:
        vo_i = vo_j = np.empty(0, dtype=int)
        vo_delta = np.empty((0, 2))

    n_ancore = n * 4  # theta, log_s, tx, ty per ogni frame
    n_coppie = len(pair_constraints) * 8
    n_vo = len(vo_pairs) * 2

    def residuals(params: np.ndarray) -> np.ndarray:
        p = params.reshape(n, 4)
        Ms = _build_matrices(params, n)

        # Ancore. La differenza angolare va normalizzata, altrimenti un giro completo
        # verrebbe contato come errore enorme invece che come nessun errore.
        d_theta = p[:, 2] - seed[:, 2]
        d_theta = np.arctan2(np.sin(d_theta), np.cos(d_theta))
        ancore = np.empty((n, 4), dtype=np.float64)
        ancore[:, 0] = d_theta * raggio * yaw_weight
        ancore[:, 1] = (p[:, 3] - seed[:, 3]) * raggio * scale_weight
        ancore[:, 2] = (p[:, 0] - seed[:, 0]) * seed_weight
        ancore[:, 3] = (p[:, 1] - seed[:, 1]) * seed_weight
        blocchi = [ancore.reshape(-1)]

        if len(idx_i):
            pts_i = np.einsum("kab,cb->kca", Ms[idx_i], corners)[:, :, :2]
            pts_j = np.einsum("kab,kcb->kca", Ms[idx_j], angoli_in_j)[:, :, :2]
            blocchi.append(((pts_i - pts_j) * pesi_coppie).reshape(-1))

        if len(vo_i):
            centri = np.einsum("nab,b->na", Ms, centro)[:, :2]
            blocchi.append(
                ((centri[vo_j] - centri[vo_i] - vo_delta) * vo_weight).reshape(-1)
            )

        return np.concatenate(blocchi)

    # Sparsita': ogni residuo dipende solo dai parametri dei frame che coinvolge.
    jac = lil_matrix((n_ancore + n_coppie + n_vo, n * 4), dtype=np.uint8)
    for i in range(n):
        jac[i * 4 : (i + 1) * 4, i * 4 : (i + 1) * 4] = 1
    for k in range(len(idx_i)):
        riga = n_ancore + k * 8
        jac[riga : riga + 8, idx_i[k] * 4 : idx_i[k] * 4 + 4] = 1
        jac[riga : riga + 8, idx_j[k] * 4 : idx_j[k] * 4 + 4] = 1
    for k in range(len(vo_i)):
        riga = n_ancore + n_coppie + k * 2
        jac[riga : riga + 2, vo_i[k] * 4 : vo_i[k] * 4 + 4] = 1
        jac[riga : riga + 2, vo_j[k] * 4 : vo_j[k] * 4 + 4] = 1

    esito = least_squares(
        residuals,
        x0,
        jac_sparsity=jac.tocsr(),
        method="trf",
        max_nfev=max_iter,
        verbose=verbose,
        xtol=1e-7,
        ftol=1e-7,
    )

    return [_compose(*esito.x[i * 4 : (i + 1) * 4]) for i in range(n)]


def vo_constraints_from_positions(
    pose_indices: list[int],
    positions_m: np.ndarray,
    gsd_canvas: float,
) -> list[tuple[int, int, float, float]]:
    """Vincoli odometrici fra scatti consecutivi fra quelli che hanno una posa.

    Gli indici sono posizioni dentro `pose_indices`, non dentro i record. Fra due scatti
    consecutivi possono esserci frame in virata, saltati perche' non entrano nel mosaico:
    lo spostamento e' comunque quello fra le due posizioni integrate, quindi la catena
    attraversa la virata senza bisogno di dare una posa agli scatti che ci stanno dentro.
    """
    vincoli = []
    for k in range(len(pose_indices) - 1):
        a, b = pose_indices[k], pose_indices[k + 1]
        de, dn = positions_m[b] - positions_m[a]
        vincoli.append((k, k + 1, de / gsd_canvas, -dn / gsd_canvas))
    return vincoli
