"""Verifica i prerequisiti della pipeline e li genera se mancanti.

Per i file generabili automaticamente (calibrazione, metadati IMU, scale,
immagini non distorte) richiama le funzioni esposte da `preprocessing/`.
Per gli input "foglia" non producibili (immagini raw del drone, exiftool,
foto di scacchiera) solleva un'eccezione esplicita con istruzioni.

Le funzioni di preprocessing sono importate via path manipulation
(`sys.path`) perché il package `preprocessing/` ha import non relativi
fra i suoi moduli (es. `from extract_xmp import ...` in `create_calib.py`).
"""
import sys

from .timing import timed


@timed
def verify_or_build(cfg):
    """Punto d'ingresso. Verifica/genera tutti i prerequisiti, in ordine di dipendenze."""
    _ensure_raw_inputs(cfg)
    _ensure_calibration(cfg)
    _ensure_metadata(cfg)
    _ensure_undistorted_images(cfg)


def _add_preprocessing_to_path(project_root):
    p = str(project_root / "preprocessing")
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_raw_inputs(cfg):
    """Verifica gli input non auto-generabili. Solleva FileNotFoundError se mancano."""
    raw_dir = cfg.resolve(cfg.raw_images_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError(
            f"Cartella immagini raw non trovata: {raw_dir}\n"
            f"Imposta `cfg.raw_images_dir` alla sottocartella DJI corretta."
        )
    raw_files = list(raw_dir.glob("*.JPG")) + list(raw_dir.glob("*.jpg"))
    if not raw_files:
        raise FileNotFoundError(f"Nessuna immagine .JPG/.jpg in {raw_dir}")

    exiftool = cfg.resolve(cfg.exiftool_path)
    if not exiftool.is_file():
        raise FileNotFoundError(
            f"exiftool non trovato in: {exiftool}\n"
            f"Scarica exiftool e imposta `cfg.exiftool_path`."
        )

    if cfg.calibration_method == "chessboard":
        cb_dir = cfg.resolve(cfg.chessboard_dir)
        if not cb_dir.is_dir() or not list(cb_dir.glob("*.jpg")):
            raise FileNotFoundError(
                f"Cartella scacchiera non trovata o vuota: {cb_dir}\n"
                f"Necessaria perché `cfg.calibration_method = 'chessboard'`."
            )


def _ensure_calibration(cfg):
    """Genera data/calibration.txt + dist.txt se mancanti."""
    data_dir = cfg.resolve("data")
    calib = data_dir / "calibration.txt"
    dist = data_dir / "dist.txt"
    if calib.exists() and dist.exists():
        return

    print(f"[bootstrap] Genero la calibrazione (metodo: {cfg.calibration_method})...")
    _add_preprocessing_to_path(cfg.project_root)

    if cfg.calibration_method == "chessboard":
        from create_calib_scacchiera import calibra_da_scacchiera
        calibra_da_scacchiera(
            str(cfg.resolve(cfg.chessboard_dir)),
            str(data_dir),
        )
    else:  # 'xmp'
        from create_calib import genera_calibrazione
        raw_dir = cfg.resolve(cfg.raw_images_dir)
        ref_image = sorted(list(raw_dir.glob("*.JPG")) + list(raw_dir.glob("*.jpg")))[0]
        genera_calibrazione(str(ref_image), str(data_dir))


def _ensure_metadata(cfg):
    """Genera data/metadati.txt se mancante."""
    metadati = cfg.resolve("data/metadati.txt")
    if metadati.exists():
        return

    print("[bootstrap] Estraggo metadati EXIF/XMP via exiftool...")
    _add_preprocessing_to_path(cfg.project_root)
    from extract_metadata import estrai_metadati_da_immagini
    estrai_metadati_da_immagini(
        str(cfg.resolve(cfg.raw_images_dir)),
        str(metadati),
        str(cfg.resolve(cfg.exiftool_path)),
    )


def _ensure_undistorted_images(cfg):
    """Genera la cartella delle immagini non distorte se assente o vuota."""
    out_dir = cfg.resolve(cfg.images_glob).parent
    if out_dir.is_dir() and len(list(out_dir.glob("*.jpg"))) > 0:
        return

    print("[bootstrap] Correzione distorsione delle immagini raw...")
    _add_preprocessing_to_path(cfg.project_root)
    from undistort_image import correggi_distorsione_cartella
    correggi_distorsione_cartella(
        str(cfg.resolve(cfg.raw_images_dir)),
        str(out_dir),
        str(cfg.resolve(cfg.exiftool_path)),
    )
