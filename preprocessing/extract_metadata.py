import glob
import json
import os
import subprocess

TAGS = [
    "-GPSLatitude", "-GPSLongitude", "-GPSAltitude",
    "-RelativeAltitude",   # XMP DJI: altezza in metri rispetto al decollo
    "-GimbalYawDegree", "-GimbalPitchDegree", "-GimbalRollDegree",
    "-FlightYawDegree", "-FlightPitchDegree", "-FlightRollDegree",
    "-FlightXSpeed", "-FlightYSpeed", "-FlightZSpeed",
    "-DateTimeOriginal",
]


def estrai_metadati_da_immagini(image_folder, output_file, exiftool_path, chunk_size=10):
    """Estrae i metadati EXIF/DJI da tutte le .jpg di `image_folder`.

    Salva il risultato come JSON array (un dict per immagine) in `output_file`.
    `chunk_size` controlla quante immagini exiftool processa per invocazione:
    serve a non superare il limite di lunghezza della command line di Windows.
    """
    immagini = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))
    if not immagini:
        raise FileNotFoundError(f"Nessuna .jpg in {image_folder}")

    metadati = []
    for i in range(0, len(immagini), chunk_size):
        chunk = immagini[i:i + chunk_size]
        result = subprocess.run(
            [exiftool_path, "-G", "-json", *TAGS, *chunk],
            capture_output=True, text=True, check=True,
        )
        metadati.extend(json.loads(result.stdout))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metadati, f, indent=2)

    print(f"Metadati estratti e salvati in {output_file}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    drone = "DJI_202604161249_001_UgCS-Create-Area-Route3"
    image_folder = os.path.join(project_root, "immagini", "immagini_drone", drone)
    output_file = os.path.join(project_root, "data", "metadati.txt")
    exiftool_path = os.path.join(project_root, "exiftool-13.53_64", "exiftool.exe")

    estrai_metadati_da_immagini(image_folder, output_file, exiftool_path)
