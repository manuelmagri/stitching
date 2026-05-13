import xml.etree.ElementTree as ET

# Funzione per leggere i dati XMP da un file immagine
def analizza_xmp(file_path):
    # Legge il file binario e trova la sezione XMP
    with open(file_path, 'rb') as file:
        data = file.read()
        start_xmp = data.find(b"<x:xmpmeta")  # Trova l'inizio dei dati XMP
        end_xmp = data.find(b"</x:xmpmeta>") + len(b"</x:xmpmeta>")  # Trova la fine
        if start_xmp == -1 or end_xmp == -1:
            raise ValueError("Dati XMP non trovati nel file")
        xmp_data = data[start_xmp:end_xmp]
        return xmp_data.decode('utf-8')  # Decodifica in stringa

# Funzione per analizzare gli elementi XML e filtrare quelli relativi al dewarping
def estrai_dati_dewarping(xmp_string):
    root = ET.fromstring(xmp_string)
    dewarping_data = {}

    # Itera su tutti i nodi XML
    for child in root.iter():
        # Filtra i tag e gli attributi rilevanti
        if "LensDistortion" in child.attrib:
            dewarping_data['LensDistortion'] = float(child.attrib["LensDistortion"])
        if "RadialDistortion" in child.attrib:
            # Estrai valori come lista di float
            radial = list(map(float, child.attrib["RadialDistortion"].split(',')))
            dewarping_data['RadialDistortion'] = radial
        if "LensCorrectionEnabled" in child.attrib:
            dewarping_data['LensCorrectionEnabled'] = child.attrib["LensCorrectionEnabled"].lower() == "true"

    return dewarping_data

# Percorso del file immagine
file_path = "C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/immagini/immagini_drone/DJI_202406171323_001_Arezzo/DJI_20240617133154_0001.JPG"

# Esecuzione del programma
try:
    # Estrai i dati XMP
    xmp_string = analizza_xmp(file_path)
    print("Dati XMP estratti:")
    print(xmp_string)

    # Analizza i dati di dewarping
    dati_dewarping = estrai_dati_dewarping(xmp_string)
    if dati_dewarping:
        print("Dati di dewarping estratti:")
        for chiave, valore in dati_dewarping.items():
            print(f"{chiave}: {valore}")
    else:
        print("Nessun dato di dewarping trovato.")
except Exception as e:
    print(f"Errore: {e}")

