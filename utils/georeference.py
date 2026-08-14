"""Collocazione del mosaico nel mondo: l'unico punto in cui entra il GPS.

Lo stitching lavora nel frame locale di `utils.localframe`, che ha gia' scala metrica
(barometro) e orientamento (bussola). Restano indeterminati due gradi di liberta', la
posizione assoluta, ed e' esattamente quello che il GPS fornisce qui.

Il fit e' una similarita' completa, non una sola traslazione, per due ragioni. La prima e'
che deve assorbire la convergenza del meridiano: la bussola misura il nord VERO, UTM usa
il nord GRIGLIA, e al sito di prova differiscono di 1,073 gradi -- 5,7 m ai bordi di un
volo di 297 m. Una georeferenziazione a sola traslazione lascerebbe il mosaico storto di
quel tanto, senza nessun sintomo evidente. La seconda e' che assorbe l'errore di scala
residuo di barometro e odometria, misurato allo 0,39% sul volo di prova.

Applicare una similarita' globale a tutte le pose non reintroduce il GPS nello stitching:
sposta, ruota e scala il risultato in blocco, e non puo' cambiarne la forma. In cambio il
residuo del fit e' una misura indipendente della qualita' della ricostruzione, perche' il
GPS non ha contribuito a produrla.
"""
import math

import numpy as np

from utils.geodesy import meridian_convergence_deg


def similarity_umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Similarita' 3x3 ai minimi quadrati che porta i punti `src` sui `dst`.

    Rotazione, scala isotropa e traslazione, senza riflessioni.
    """
    if len(src) < 2:
        raise ValueError("servono almeno due punti per stimare una similarita'")

    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    S, D = src - mu_s, dst - mu_d

    U, valori, Vt = np.linalg.svd(D.T @ S / len(src))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    varianza = float((S**2).sum() / len(src))
    scala = float(valori.sum() / varianza) if varianza > 0 else 1.0
    t = mu_d - scala * (R @ mu_s)

    A = np.eye(3, dtype=np.float64)
    A[:2, :2] = scala * R
    A[:2, 2] = t
    return A


def expected_rotation_deg(lat: float, lon: float, grid_north: bool) -> float:
    """Quanta rotazione il fit dovrebbe assorbire, dato com'e' orientato il frame locale.

    Un frame costruito sulla bussola e' in nord VERO e deve ancora scontare la convergenza
    del meridiano; uno costruito direttamente in UTM e' gia' in nord griglia e non deve
    scontare nulla. Il segno meno e' il verso della y del canvas.
    """
    return 0.0 if grid_north else -meridian_convergence_deg(lat, lon)


def gps_targets_px(
    records: list[dict],
    to_utm,
    gsd_canvas: float,
) -> tuple[np.ndarray, tuple[float, float]]:
    """Dove il GPS vorrebbe i centri dei frame, in pixel di canvas, e l'origine UTM.

    L'origine e' la posizione UTM del primo record: da li' in poi il pixel (x, y) del
    canvas corrisponde a UTM (est0 + x*gsd, nord0 - y*gsd), con y verso sud.
    """
    utm = np.array([to_utm.transform(r["lon"], r["lat"]) for r in records], dtype=np.float64)
    origine = (float(utm[0, 0]), float(utm[0, 1]))
    target = np.empty_like(utm)
    target[:, 0] = (utm[:, 0] - origine[0]) / gsd_canvas
    target[:, 1] = -(utm[:, 1] - origine[1]) / gsd_canvas
    return target, origine


def pose_centers_px(transforms: list[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    """Centro immagine di ogni posa, in pixel di canvas."""
    w, h = image_size
    centro = np.array([w / 2.0, h / 2.0, 1.0], dtype=np.float64)
    return np.array([(M @ centro)[:2] for M in transforms], dtype=np.float64)


def fit_to_gps(
    centers_px: np.ndarray,
    targets_px: np.ndarray,
    gsd_canvas: float,
    expected_rotation_deg: float,
    outlier_sigma: float = 4.0,
) -> tuple[np.ndarray, dict]:
    """Similarita' 3x3 dal canvas locale al canvas allineato a UTM, piu' la diagnostica.

    `expected_rotation_deg` e' quanta rotazione ci si aspetta che il fit debba assorbire,
    e serve solo da controllo: il residuo fra questa e quella misurata dice se il frame
    locale era orientato come si credeva. Chi costruisce il frame sulla bussola misura il
    nord VERO e deve aspettarsi la convergenza del meridiano (col segno cambiato, perche'
    il canvas ha y verso sud, quindi una rotazione oraria nel mondo appare antioraria
    qui); chi lo costruisce direttamente in UTM e' gia' in nord griglia e deve aspettarsi
    zero.

    Una seconda passata esclude i fix GPS che si discostano oltre `outlier_sigma` deviazioni
    robuste, cosi' un singolo fix sbagliato non trascina l'intero mosaico.
    """
    A = similarity_umeyama(centers_px, targets_px)

    def residui(A):
        proiettati = (np.column_stack([centers_px, np.ones(len(centers_px))]) @ A.T)[:, :2]
        return np.linalg.norm(proiettati - targets_px, axis=1)

    scarti = residui(A)
    tenuti = np.ones(len(scarti), dtype=bool)
    if len(scarti) > 8:
        mediana = float(np.median(scarti))
        mad = float(np.median(np.abs(scarti - mediana))) * 1.4826
        if mad > 0:
            tenuti = scarti < mediana + outlier_sigma * mad
            if tenuti.sum() >= max(8, int(0.5 * len(scarti))):
                A = similarity_umeyama(centers_px[tenuti], targets_px[tenuti])
                scarti = residui(A)
            else:
                tenuti = np.ones(len(scarti), dtype=bool)

    rotazione = math.degrees(math.atan2(A[1, 0], A[0, 0]))
    scala = float(math.hypot(A[0, 0], A[1, 0]))

    info = {
        "rotation_deg": rotazione,
        "scale": scala,
        "expected_rotation_deg": expected_rotation_deg,
        "rotation_residual_deg": rotazione - expected_rotation_deg,
        "rms_m": float(np.sqrt((scarti[tenuti] ** 2).mean()) * gsd_canvas),
        "median_m": float(np.median(scarti[tenuti]) * gsd_canvas),
        "max_m": float(scarti[tenuti].max() * gsd_canvas),
        "outliers": int((~tenuti).sum()),
        "n": int(len(scarti)),
    }
    return A, info


def apply_to_poses(A: np.ndarray, transforms: list[np.ndarray]) -> list[np.ndarray]:
    """Compone la similarita' globale davanti a ogni posa."""
    return [A @ M for M in transforms]
