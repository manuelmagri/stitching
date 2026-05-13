"""Pipeline di stitching drone: moduli di supporto al main.

Sezioni:
  io_loader        - calibrazione, metadati EXIF/IMU, immagini in streaming
  gps_utm          - conversione lat/lon -> UTM e Delta posizione metrica
  feature          - ORB con suddivisione in nxn quadranti
  matching         - FLANN/Lowe/RANSAC tra due frame
  vo               - visual odometry frame-by-frame
  pose_fusion      - VO + prior GPS/IMU, fallback in curva
  homography_curve - omografia analitica per i frame in curva
  mosaic           - warp incrementale su canvas pre-allocato in UTM
  geotiff_writer   - output GeoTIFF UTM apribile in QGIS
"""
