"""Pacchetto utilita' per VO + stitching + ortomosaico georeferenziato.

Layout:
    config            -> Config dataclass (parametri pipeline)
    io_calibration    -> caricamento calibrazione camera (K, distortion, projMatrix)
    io_metadata       -> parsing metadati.txt + file IMU derivati
    geo               -> WGS84 -> UTM, allineamento Umeyama VO->GPS
    passes            -> segmentazione del percorso "alla greca" in passate
    features          -> ORB con detection in griglia n x n
    matching          -> FLANN + Lowe ratio + RANSAC (omografia / matrice essenziale)
    vo                -> visual odometry frame-a-frame (essential matrix)
    stitch_chunk      -> cv2.Stitcher per passata
    compose           -> composizione chunk su canvas UTM (con match GPS-aware tra passate)
    ortho             -> scrittura mosaico georeferenziato (worldfile + .prj)
    pipeline          -> orchestrazione end-to-end (entry point: pipeline.run(cfg))
"""
