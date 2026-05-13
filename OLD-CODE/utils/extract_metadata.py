import os
import subprocess
import glob


def extract_metadata_from_images(image_folder, output_file, chunk_size=10):
    """
    Estrae i dati IMU da tutte le immagini in una cartella e li salva in un file di testo.

    :param image_folder: Percorso della cartella contenente le immagini.
    :param output_file: Percorso del file di output per salvare i dati IMU.
    :param chunk_size: Numero massimo di immagini da processare in un singolo comando.
    """

    exiftool_path = r"C:/Users/Asus/Desktop/exiftool-12.99_64/exiftool.exe"

    image_files = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

    # Inizializza il file di output con l'apertura del JSON
    with open(output_file, 'w') as outfile:
        outfile.write('[\n')  # Scrivi l'apertura del JSON array

    for i in range(0, len(image_files), chunk_size):
        # Prendi un blocco di immagini
        chunk = image_files[i:i + chunk_size]
        chunk_str = " ".join(f'"{img}"' for img in chunk)  # Usa virgolette per file con spazi nel nome

        # Comando per estrarre i dati IMU da ExifTool
        command = (
            f'{exiftool_path} -G -json '
            '-GPSLatitude -GPSLongitude -GPSAltitude '
            '-GimbalYawDegree -GimbalPitchDegree -GimbalRollDegree '
            '-FlightYawDegree -FlightPitchDegree -FlightRollDegree '
            '-FlightXSpeed -FlightYSpeed -FlightZSpeed -DateTimeOriginal '
            f'{chunk_str}'
        )

        # Esegui il comando e cattura l'output
        try:
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                # Aggiungi l'output JSON al file di testo
                with open(output_file, 'a') as outfile:
                    # Rimuovi le parentesi quadre del JSON prodotto da ExifTool e aggiungi una virgola alla fine
                    json_output = result.stdout.strip()
                    if json_output.startswith("[") and json_output.endswith("]"):
                        json_output = json_output[1:-1]  # Rimuovi la prima e l'ultima parentesi quadra

                    # Aggiungi una virgola solo se non è l'ultimo chunk
                    outfile.write(json_output)
                    if i + chunk_size < len(image_files):
                        outfile.write(",\n")  # Virgola solo tra i chunk
            else:
                print(f"Errore nell'estrazione dei metadati per il blocco {i // chunk_size + 1}: {result.stderr}")
        except Exception as e:
            print(f"Si è verificato un errore con il blocco {i // chunk_size + 1}: {str(e)}")

    # Chiudi il file JSON
    with open(output_file, 'a') as outfile:
        outfile.write('\n]')  # Scrivi la chiusura del JSON array

    print(f"Metadati estratti e salvati in {output_file}")


# Definisci il percorso della cartella delle immagini e del file di output
image_folder = r"C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/immagini/immagini_drone/DJI_202406171516_002_Arezzo"
output_file = r"C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/data/metadati.txt"

# Estrai i dati IMU con un blocco massimo di 10 immagini per volta
extract_metadata_from_images(image_folder, output_file, chunk_size=10)
