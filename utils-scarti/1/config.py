"""Configurazione della pipeline. Una sola dataclass `Config`, modificabile da main.py."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DebugConfig:
    save_features: bool = False     # salva immagini con keypoints ORB
    save_matches: bool = False      # salva immagini dei match tra frame
    save_chunks: bool = True        # salva il png di ogni chunk-passata
    verbose: bool = True


@dataclass
class Config:
    project_root: Path

    # ---- range immagini ---------------------------------------------------
    # (0, 0) = tutte le immagini; altrimenti [start, end) sull'elenco ordinato
    start: int = 0
    end: int = 0

    # ---- detection feature (ORB) -----------------------------------------
    n_features_total: int = 4000   # budget complessivo per immagine
    n_grid: int = 5                # detection in n x n quadranti

    # ---- matching (FLANN-LSH + Lowe + RANSAC) ----------------------------
    lowe_ratio: float = 0.7
    ransac_thresh_px: float = 3.0
    min_matches: int = 12

    # ---- VO (matrice essenziale) -----------------------------------------
    essential_prob: float = 0.999
    essential_thresh_px: float = 0.5
    use_imu_scales: bool = True    # se True, modula la traslazione VO con scales.txt

    # ---- Stitcher per passata --------------------------------------------
    stitcher_confidence_thresh: float = 0.3
    stitcher_resol_factor: float = 0.6   # downscale interno usato da Stitcher

    # ---- Pass segmentation (percorso "alla greca") -----------------------
    pass_yaw_turn_threshold_deg: float = 90.0
    pass_min_len: int = 5

    # ---- Composizione e match GPS-aware tra passate ----------------------
    inter_pass_search_radius_m: float = 25.0   # raggio in metri per accoppiare frame tra passate
    canvas_resolution_m_per_px: float = 0.05    # GSD canvas finale (m/px). 5 cm/px default

    # ---- Resize lettura immagini per VO/feature -------------------------
    image_downscale_for_vo: float = 2.5

    # ---- Altitudine AGL stimata (m). Se None la calcoliamo dai dati ------
    assumed_agl_m: float | None = None

    # ---- Debug ------------------------------------------------------------
    debug: DebugConfig = field(default_factory=DebugConfig)

    # ---- Paths derivate --------------------------------------------------
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def images_dir(self) -> Path:
        return self.project_root / "immagini" / "immagini_drone" / "immagini_senza_distorsione"

    @property
    def output_dir(self) -> Path:
        d = self.project_root / "output"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def metadati_file(self) -> Path:
        return self.data_dir / "metadati.txt"

    @property
    def scales_file(self) -> Path:
        return self.data_dir / "scales.txt"

    @property
    def yaws_file(self) -> Path:
        return self.data_dir / "yaws.txt"
