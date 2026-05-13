"""Mosaico aereo con cv2.Stitcher in modalità SCANS.

SCANS è il preset OpenCV pensato per scansioni planari (drone aerial mapping).
Internamente combina:
  - ORB feature detection;
  - AffineBestOf2NearestMatcher (matching pairwise *e* cross-strip);
  - bundle adjustment affine partial (4 DoF: rot + scala + traslazione);
  - graph-cut seam finder;
  - multi-band blending.

È la pipeline canonica per questo problema, tutto in C++ ottimizzato.
"""
import cv2

from .timing import timed


_STATUS_MESSAGES = {
    cv2.Stitcher_OK: "OK",
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "servono più immagini con sufficiente overlap",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "stima omografia/affine fallita",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "bundle adjustment fallito",
}


@timed
def render_mosaic(images, confidence_thresh=None):
    """Costruisce il mosaico finale dalla sequenza di immagini.

    Args:
        images: lista di immagini BGR (np.uint8) ordinate.
        confidence_thresh: soglia per accettare una coppia di immagini come
            parte dello stesso panorama. `None` lascia il default di OpenCV
            (1.0). Abbassare a 0.3-0.5 se Stitcher scarta troppi frame perché
            i match cross-strip hanno bassa confidenza.

    Ritorna l'immagine del mosaico (np.uint8 BGR).
    Lancia `RuntimeError` se Stitcher fallisce.
    """
    stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
    if confidence_thresh is not None:
        stitcher.setPanoConfidenceThresh(float(confidence_thresh))

    status, mosaic = stitcher.stitch(images)
    if status != cv2.Stitcher_OK:
        msg = _STATUS_MESSAGES.get(status, f"codice {status}")
        raise RuntimeError(f"cv2.Stitcher fallito: {msg}")
    return mosaic
