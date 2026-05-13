"""Configurazione centralizzata della pipeline di stitching + VO."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DebugConfig:
    """Flag e percorsi per gli output di debug. Tutto opzionale e disattivabile."""
    save_features: bool = False
    features_dir: str = "immagini/immagini_drone/feature"


@dataclass
class Config:
    """Parametri della pipeline. Si costruisce con `Config(project_root=...)` e si modificano gli attributi prima di passarlo a `pipeline.run`."""
    project_root: Path

    # I/O (percorsi relativi a project_root)
    images_glob: str = "immagini/immagini_drone/immagini_senza_distorsione/*.jpg"
    calibration_file: str = "data/calibration.txt"
    scales_file: str = "data/scales.txt"
    output_dir: str = "output"

    # Input non auto-generabili (usati dal bootstrap per generare i file mancanti)
    raw_images_dir: str = "immagini/immagini_drone/DJI_202604161249_001_UgCS-Create-Area-Route3"
    exiftool_path: str = "exiftool-13.53_64/exiftool.exe"
    calibration_method: str = "xmp"  # 'xmp' (singola immagine) | 'chessboard' (scacchiera)
    chessboard_dir: str = "immagini/immagini_scacchiera_drone"

    # Range immagini: start=end=0 → tutte; start>0, end=0 → da start fino alla fine
    start: int = 0
    end: int = 0

    # Preprocessing
    resize_factor: float = 2.5

    # Feature detection per la VO (ORB su griglia n_grid × n_grid)
    n_grid: int = 4
    n_features_total: int = 2000
    quadrant_overlap: int = 40

    # Matching per la VO
    lowe_ratio: float = 0.7
    min_good_matches: int = 10

    # RANSAC per la matrice essenziale (VO)
    essential_ransac_thr: float = 0.5
    essential_ransac_prob: float = 0.999

    # Metodo di composizione del mosaico:
    #   "ortho"  → ortomosaico geo-riferito da GPS + gimbal yaw + AGL auto-cal.
    #              Funziona se il gimbal pitch è ~ -90° (nadir).
    #   "scans"  → cv2.Stitcher_SCANS (legacy). Drift su sequenze lunghe.
    mosaic_method: str = "ortho"

    # Override AGL (m). None → auto-calibrato dalle feature in crociera.
    ortho_agl_m: Optional[float] = None
    # Ground sampling distance del canvas (m/pixel). None → AGL/focale (no upscaling).
    ortho_gsd: Optional[float] = None

    # cv2.Stitcher (modalità SCANS): soglia di confidenza per accettare una
    # coppia come parte dello stesso panorama. None → default OpenCV (1.0).
    # Abbassare (es. 0.3) se Stitcher scarta troppi frame.
    stitcher_confidence_thresh: Optional[float] = None

    # OpenCV può usare OpenCL via UMat per accelerare. Su alcune GPU/driver
    # genera CL_OUT_OF_RESOURCES sui buffer grandi del blender; lasciare a
    # False se non hai motivo specifico per attivarlo.
    use_opencl: bool = False

    debug: DebugConfig = field(default_factory=DebugConfig)

    def resolve(self, relative_path: str) -> Path:
        """Restituisce il path assoluto di un percorso relativo a project_root."""
        return Path(self.project_root) / relative_path
