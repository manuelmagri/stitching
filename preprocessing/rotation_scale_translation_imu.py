import numpy as np
import json
import os
from datetime import datetime

PROJECT_ROOT : str = os.path.dirname(os.path.dirname(__file__))

# Calcola la matrice di rotazione a partire dagli angoli di rollio, beccheggio e imbardata (in gradi)
def calculate_rotation_matrix(roll, pitch, yaw):
    # Converti gli angoli da gradi a radianti
    roll_rad = np.radians(roll)
    pitch_rad = np.radians(pitch)
    yaw_rad = np.radians(yaw)

    # Matrice di rotazione attorno all'asse X (Roll)
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll_rad), -np.sin(roll_rad)],
                    [0, np.sin(roll_rad), np.cos(roll_rad)]])

    # Matrice di rotazione attorno all'asse Y (Pitch)
    R_y = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                    [0, 1, 0],
                    [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])

    # Matrice di rotazione attorno all'asse Z (Yaw)
    R_z = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                    [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                    [0, 0, 1]])

    # Matrice di rotazione complessiva applicando prima R_x, poi R_y e infine R_z
    R = R_z @ R_y @ R_x
    return R

# Calcola il fattore di scala basandosi sulla velocità media tra due frame e il tempo trascorso
def calculate_scale(flight_speed1, flight_speed2, delta_time):
    avg_speed1 = np.linalg.norm(flight_speed1)  # Modulo della velocità iniziale
    avg_speed2 = np.linalg.norm(flight_speed2)  # Modulo della velocità finale
    avg_speed = (avg_speed1 + avg_speed2) / 2  # Velocità media
    scale = avg_speed * delta_time  # Scala = velocità media * tempo trascorso
    return scale

# Calcola il vettore di traslazione usando velocità e tempo
def calculate_translation(flight_speed, delta_time):
    translation = flight_speed * delta_time  # Traslazione = velocità * tempo
    return translation

# Estrae i metadati da un file JSON
def extract_metadata(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)  # Carica i dati JSON in un dizionario
    return data

# Scrive su file le matrici di rotazione, scale, traslazioni, timestamp e yaw
def write_rotations_scales_and_translations(rotations, scales, translations, timestamps, yaws, rotation_file, scale_file, translation_file, timestamp_file, yaws_file):
    with open(rotation_file, 'w') as file:
        for idx, rotation in enumerate(rotations):
            file.write(f"Rotation Matrix {idx + 1}:\n{rotation}\n\n")

    with open(scale_file, 'w') as file:
        for idx, scale in enumerate(scales):
            file.write(f"Scale {idx + 1}: {scale}\n")

    with open(translation_file, 'w') as file:
        for idx, translation in enumerate(translations):
            file.write(f"Translation Vector {idx + 1}: {translation}\n")

    with open(timestamp_file, 'w') as file:
        for idx, timestamp in enumerate(timestamps):
            file.write(f"Timestamp Vector {idx + 1}: {timestamp}\n")

    with open(yaws_file, 'w') as file:
        for idx, yaw in enumerate(yaws):
            file.write(f"Yaw {idx + 1}: {yaw}\n")

# Funzione principale che esegue il calcolo delle pose a partire dai metadati
def main():
    metadata = extract_metadata(os.path.join(PROJECT_ROOT, "data/metadati.txt"))

    rotations = []
    scales = []
    translations = []  # Lista per i vettori di traslazione
    speeds = []
    timestamps = []
    yaws = []

    for entry in metadata:
        roll = float(entry["XMP:FlightRollDegree"])  # Angolo di rollio
        pitch = float(entry["XMP:FlightPitchDegree"])  # Angolo di beccheggio
        yaw = float(entry["XMP:FlightYawDegree"])  # Angolo di imbardata
        speed = np.array([
            float(entry["XMP:FlightXSpeed"]),
            float(entry["XMP:FlightYSpeed"]),
            float(entry["XMP:FlightZSpeed"])
        ])  # Vettore velocità

        yaws.append(yaw)
        timestamp_str = str(entry["EXIF:DateTimeOriginal"])  # Estrazione timestamp
        timestamp = datetime.strptime(timestamp_str, "%Y:%m:%d %H:%M:%S")

        # Calcola la matrice di rotazione e la memorizza
        rotation = calculate_rotation_matrix(roll, pitch, yaw)
        rotations.append(rotation)

        if len(scales) > 0:
            delta_time = (timestamp - timestamps[-1]).total_seconds()  # Differenza di tempo
            scale = calculate_scale(speed, speeds[-1], delta_time)  # Calcolo della scala
            scales.append(scale)
            translation = calculate_translation(speed, delta_time)  # Calcolo traslazione
            translations.append(translation)
        else:
            delta_time = 1  # Se primo frame, assegna un delta_time arbitrario
            scale = np.linalg.norm(speed) * delta_time
            scales.append(scale)
            translation = calculate_translation(speed, delta_time)
            translations.append(translation)

        timestamps.append(timestamp)
        speeds.append(speed)

    # Scrivi i risultati su file
    write_rotations_scales_and_translations(rotations, scales, translations, timestamps, yaws,
                                            os.path.join(PROJECT_ROOT, 'data/rotations.txt'),
                                            os.path.join(PROJECT_ROOT, 'data/scales.txt'),
                                            os.path.join(PROJECT_ROOT, 'data/translations.txt'),
                                            os.path.join(PROJECT_ROOT, 'data/timestamps.txt'),
                                            os.path.join(PROJECT_ROOT, 'data/yaws.txt'))

if __name__ == '__main__':
    main()
