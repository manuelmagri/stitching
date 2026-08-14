"""Geometria del volo: scatti in virata e segmentazione in passate.

Tutto qui dentro usa solo l'assetto IMU (bussola e gimbal), mai il GPS.

Le passate di un lawnmower sono percorse a greca, quindi la bussola assume due valori
dominanti a 180 gradi l'uno dall'altro (sul volo di prova 43 e -137). Da qui la
segmentazione: si stima l'asse dominante e si classifica ogni scatto in base a quanto se
ne discosta.

Due criteri usati in passato sono stati scartati, entrambi dopo averli misurati sul volo
di prova:

- La VELOCITA' di imbardata scambiava per virata ogni oscillazione momentanea della
  bussola. Il confronto con la direzione dominante e' insensibile alle oscillazioni,
  perche' guarda quanto ci si e' allontanati e non quanto in fretta.

- L'inclinazione del DRONE (`flight_roll_deg`) non dice nulla sull'immagine, perche' il
  gimbal e' stabilizzato: `gimbal_pitch_deg` ha mediana -90,00 e solo 28 scatti su 835
  se ne discostano oltre 2 gradi. Il drone intanto vola con un roll di crociera di 5,2
  gradi di mediana, quindi una soglia assoluta a 8 gradi scattava sul rumore e spezzava
  le passate a meta'. Cio' che conta e' dove guarda la camera, non come sta il velivolo:
  per questo il secondo criterio qui e' lo scarto del GIMBAL dal nadir.

Lo scarto di rotta e' nettamente bimodale -- 745 scatti entro 2 gradi dall'asse, poi il
vuoto fino a 20 -- quindi la tolleranza non e' un parametro critico: qualunque valore fra
5 e 20 gradi produce la stessa segmentazione.
"""
import math

import numpy as np


def _yaw_diff(a: float, b: float) -> float:
    """Differenza minima fra due angoli in gradi, normalizzata a [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def dominant_axis_deg(records: list[dict]) -> float:
    """Direzione dominante delle passate, modulo 180 gradi.

    Raddoppiando gli angoli, due rotte opposte diventano lo stesso angolo, quindi la media
    vettoriale degli angoli raddoppiati da' l'asse del volo senza farsi confondere dal
    verso di percorrenza.
    """
    if not records:
        return 0.0

    doppi = np.radians([r["flight_yaw_deg"] for r in records]) * 2.0
    return math.degrees(np.angle(np.mean(np.exp(1j * doppi)))) / 2.0


def mark_curves(
    records: list[dict],
    yaw_tolerance_deg: float = 15.0,
    nadir_tolerance_deg: float = 3.0,
) -> dict:
    """Aggiunge in-place `is_curve` e `heading` a ogni record. Ritorna statistiche.

    Uno scatto e' in virata se la sua rotta si discosta da entrambe le direzioni dominanti
    oltre `yaw_tolerance_deg`, oppure se il gimbal non guardava il nadir entro
    `nadir_tolerance_deg` -- nel qual caso l'impronta a terra non e' il rettangolo che
    tutta la pipeline assume.

    `heading` vale +1 sugli scatti che percorrono l'asse nel verso di `axis_deg`, -1 su
    quelli che lo percorrono al contrario, ed e' None sugli scatti in virata. E' quello
    che separa due passate adiacenti, che sono contigue nel tempo ma opposte nel verso.
    """
    axis_deg = dominant_axis_deg(records)

    for record in records:
        scarto = _yaw_diff(record["flight_yaw_deg"], axis_deg)
        record["heading"] = 1 if abs(scarto) <= 90.0 else -1

        # Distanza dalla direzione dominante piu' vicina, indipendente dal verso.
        fuori_rotta = min(abs(scarto), 180.0 - abs(scarto))
        fuori_nadir = abs(record.get("gimbal_pitch_deg", -90.0) + 90.0)
        record["is_curve"] = (
            fuori_rotta > yaw_tolerance_deg or fuori_nadir > nadir_tolerance_deg
        )
        if record["is_curve"]:
            record["heading"] = None

    in_curva = sum(1 for r in records if r["is_curve"])
    return {
        "axis_deg": axis_deg,
        "curve": in_curva,
        "straight": len(records) - in_curva,
    }


def group_into_legs(
    records: list[dict], positions_m: np.ndarray, min_extent_m: float
) -> list[dict]:
    """Passate: sequenze massimali di scatti non in virata con lo stesso `heading`.

    Ogni passata e' un dict con `frames` (indici in `records`, in ordine temporale) e
    `heading`. Gli scatti in virata non appartengono a nessuna passata e non entrano nel
    mosaico: la camera si muove in fretta e la copertura e' fuori dall'area da rilevare.

    Non spariscono pero' dal problema. La loro odometria resta, ed e' quella che trasporta
    la posizione da una passata alla successiva quando il matching fotografico fra passate
    adiacenti non regge: senza GPS e' l'unica rete di sicurezza rimasta, ed e' `utils.poses`
    a usarla, sommando i delta attraverso la virata.

    Una sequenza corta a meta' di una inversione non e' una passata, e trattarla come tale
    creerebbe passate spurie con vincoli inaffidabili. La domanda giusta pero' non e'
    quanti scatti contenga -- cinque scatti su un volo da ottocento e cinque su uno da
    venti non vogliono dire la stessa cosa -- ma se COPRA TERRENO NUOVO: si tiene la
    sequenza che si estende per almeno `min_extent_m`, tipicamente l'impronta a terra di un
    singolo scatto. Cosi' il criterio e' lo stesso su qualunque volo, e non c'e' una
    costante da ritarare quando cambia la camera.
    """
    legs: list[dict] = []
    corrente: list[int] = []
    heading_corrente = None

    def chiudi():
        nonlocal corrente, heading_corrente
        if corrente:
            legs.append({"frames": corrente, "heading": heading_corrente})
        corrente = []
        heading_corrente = None

    for i, record in enumerate(records):
        if record["is_curve"]:
            chiudi()
            continue
        if corrente and record["heading"] != heading_corrente:
            chiudi()
        corrente.append(i)
        heading_corrente = record["heading"]
    chiudi()

    tenute = []
    for leg in legs:
        # Il minimo di tre scatti non e' una taratura ma una condizione di forma: due
        # scatti non sono una passata, non stabiliscono una direzione, e se fra i due c'e'
        # un salto -- un volo interrotto e ripreso altrove -- l'estensione da sola li
        # promuove a passata lunghissima. Visto succedere su un volo con un salto di 133 m.
        if len(leg["frames"]) >= 3 and leg_extent_m(positions_m, leg["frames"]) >= min_extent_m:
            tenute.append(leg)
            continue
        for i in leg["frames"]:
            records[i]["is_curve"] = True
            records[i]["heading"] = None
    return tenute


def leg_extent_m(positions_m: np.ndarray, leg_frames: list[int]) -> float:
    """Distanza fra il primo e l'ultimo scatto di una sequenza, in metri."""
    if len(leg_frames) < 2:
        return 0.0
    return float(np.linalg.norm(positions_m[leg_frames[-1]] - positions_m[leg_frames[0]]))


def subsample_leg(
    positions_m: np.ndarray,
    leg_frames: list[int],
    footprint_along_m: float,
    target_overlap: float,
) -> list[int]:
    """Sottoinsieme di `leg_frames` che mantiene circa `target_overlap` lungo la passata.

    E' la leva contro lo spreco: il volo di prova ha overlap frontale 80%, cioe' quattro
    scatti quasi identici dove ne basterebbero due. Diradare riduce nello stesso colpo le
    coppie da matchare, i parametri da ottimizzare e i frame da comporre.

    Primo e ultimo della passata si tengono sempre: sono quelli che coprono le estremita',
    che nessun altro scatto raggiunge.
    """
    if len(leg_frames) <= 2:
        return list(leg_frames)

    passo_minimo = footprint_along_m * max(1.0 - target_overlap, 0.05)
    tenuti = [leg_frames[0]]
    ultimo = positions_m[leg_frames[0]]

    for frame in leg_frames[1:-1]:
        if float(np.hypot(*(positions_m[frame] - ultimo))) >= passo_minimo:
            tenuti.append(frame)
            ultimo = positions_m[frame]

    tenuti.append(leg_frames[-1])
    return tenuti
