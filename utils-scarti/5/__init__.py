"""Pipeline di stitching drone: moduli di supporto al main.

Sezioni:
  io_loader        - calibrazione, metadati EXIF/IMU, immagini in streaming
  gps_utm          - conversione lat/lon -> UTM e Delta posizione metrica
  feature          - ORB con suddivisione in nxn quadranti
  matching         - FLANN/Lowe/RANSAC tra due frame
  vo               - visual odometry frame-by-frame (scala da Delta UTM)
  pose_fusion      - VO + prior GPS/IMU per traiettoria metrica
  mosaic           - warp incrementale su canvas pre-allocato in UTM
  geotiff_writer   - output GeoTIFF UTM apribile in QGIS
"""

from .io_loader import (
    CameraCalibration,
    FrameMeta,
    FrameStream,
    carica_calibrazione,
    carica_frames_meta,
)
from .gps_utm import (
    UtmFrame,
    UtmProjector,
    deltas_metrici,
    passi_metrici,
    proietta_frames,
    seleziona_per_overlap,
)
from .feature import (
    FeatureSet,
    detect_orb_nxn,
    draw_keypoints,
)
from .matching import (
    MatchResult,
    match_frames,
    draw_matches,
)
from .vo import (
    VoStep,
    accumulate,
    estimate_step,
    transf,
)
from .pose_fusion import (
    WorldPose,
    fuse,
    yaw_unwrap,
)
from .mosaic import (
    MosaicCanvas,
    build_mosaic,
    crop_to_content,
    paste_frame,
    plan_canvas,
)
from .geotiff_writer import scrivi_geotiff

__all__ = [
    # io_loader
    "CameraCalibration",
    "FrameMeta",
    "FrameStream",
    "carica_calibrazione",
    "carica_frames_meta",
    # gps_utm
    "UtmFrame",
    "UtmProjector",
    "deltas_metrici",
    "passi_metrici",
    "proietta_frames",
    "seleziona_per_overlap",
    # feature
    "FeatureSet",
    "detect_orb_nxn",
    "draw_keypoints",
    # matching
    "MatchResult",
    "match_frames",
    "draw_matches",
    # vo
    "VoStep",
    "accumulate",
    "estimate_step",
    "transf",
    # pose_fusion
    "WorldPose",
    "fuse",
    "yaw_unwrap",
    # mosaic
    "MosaicCanvas",
    "build_mosaic",
    "crop_to_content",
    "paste_frame",
    "plan_canvas",
    # geotiff_writer
    "scrivi_geotiff",
]
