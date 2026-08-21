"""Geometria del volo: scatti in virata, segmentazione in passate, scelta dei fotogrammi.

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

LA GRECA D'AREA NON E' L'UNICA GRECA
------------------------------------
Il volo di prova (`Create-Area-Route3`, 835 scatti a 30 m) e' una greca su un'area
compatta: 18 passate da 260 m, contro un'impronta a terra along-track di 33 m. Ogni
passata e' otto impronte, la virata avviene FUORI dall'area, e gli scatti in virata si
buttano senza perdere niente.

Sui quattro voli DJI `UgCS-VT*` la stessa greca ha proporzioni rovesciate: sono rilievi di
CORRIDOIO -- una striscia lunga da 350 a 1300 m e larga poche decine -- percorsi a zig-zag
trasversale. Misurato: passate da 15 a 40 m contro un'impronta along-track di 55-66 m,
cioe' una passata intera copre MENO di un singolo fotogramma, e le virate stanno dentro
il corridoio, sopra il terreno da rilevare, non fuori.

Ne seguono le due scelte che questo modulo fa, e che sono l'unica differenza rispetto alla
versione che funzionava solo sull'area compatta:

- `group_into_legs` non misura piu' una passata contro l'impronta e basta, perche' su un
  corridoio nessuna passata la raggiunge e il volo usciva senza nemmeno una passata
  riconosciuta. Il metro diventa la passata tipica DI QUESTO VOLO quando e' piu' corta.

- `select_frames` non sceglie piu' i fotogrammi passata per passata, ma lungo il percorso.
  Chi entra nel mosaico lo decide la copertura, non l'appartenenza a una passata: sul
  volo di prova sceglie esattamente gli stessi fotogrammi di prima, uno per uno, sui
  corridoi ne sceglie da 10 a 222 dove prima non ne entrava nessuno.
"""
import math

import numpy as np


# Quanto puo' essere piu' corta di una passata tipica una sequenza, e restare una passata.
# Non e' critica: sui corridoi le estensioni sono quantizzate dal passo fra scatti (su
# `VT3` valgono 15, 23, 31, 39 m, cioe' 2, 3, 4, 5 passi da 7,7 m), quindi qualunque valore
# fra 0,3 e 0,5 sposta 6 sequenze su 86, e sotto `select_frames` una sequenza in piu' o in
# meno vale due fotogrammi. Sul volo di prova non tocca niente: le 18 passate misurano
# 259-267 m e il minimo resta l'impronta.
FRAZIONE_PASSATA_TIPICA = 0.5


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
    """Aggiunge in-place `usable`, `is_curve` e `heading` a ogni record. Ritorna statistiche.

    Sono due domande diverse, e tenerle separate e' cio' che permette di cucire anche i voli
    di corridoio:

    `usable` e' una proprieta' dell'IMMAGINE. Se il gimbal non guardava il nadir entro
    `nadir_tolerance_deg`, l'impronta a terra non e' il rettangolo che tutta la pipeline
    assume, e quello scatto non puo' entrare nel mosaico in nessun caso. E' l'unico criterio
    che scarta davvero: sul volo di prova toglie 28 scatti su 835, sui quattro corridoi uno
    solo su 1119 -- li' il gimbal sta fra -89,9 e -90,0 gradi dal primo scatto all'ultimo.

    `is_curve` e' una proprieta' della TRAIETTORIA: lo scatto non stava percorrendo una
    passata, perche' la sua rotta si discosta da entrambe le direzioni dominanti oltre
    `yaw_tolerance_deg`. Serve a tagliare le passate, non a buttare scatti. Su un'area
    compatta le due cose coincidevano quasi sempre -- si vira fuori dall'area, quindi lo
    scatto in virata era anche inutile -- ma su un corridoio si vira SOPRA il terreno da
    rilevare, e trattare le due domande come una sola buttava via meta' del volo.

    `heading` vale +1 sugli scatti che percorrono l'asse nel verso di `axis_deg`, -1 su
    quelli che lo percorrono al contrario, ed e' None sugli scatti in virata. E' quello
    che separa due passate adiacenti, che sono contigue nel tempo ma opposte nel verso.
    """
    axis_deg = dominant_axis_deg(records)

    for record in records:
        scarto = _yaw_diff(record["flight_yaw_deg"], axis_deg)
        record["heading"] = 1 if abs(scarto) <= 90.0 else -1

        # Se la sorgente non registra l'assetto del gimbal il criterio e' inerte, ed e'
        # corretto che lo sia: su immagini gia' ortorettificate la sua premessa non vale.
        fuori_nadir = abs(record.get("gimbal_pitch_deg", -90.0) + 90.0)
        record["usable"] = fuori_nadir <= nadir_tolerance_deg

        # Distanza dalla direzione dominante piu' vicina, indipendente dal verso.
        fuori_rotta = min(abs(scarto), 180.0 - abs(scarto))
        record["is_curve"] = fuori_rotta > yaw_tolerance_deg or not record["usable"]
        if record["is_curve"]:
            record["heading"] = None

    return {
        "axis_deg": axis_deg,
        "curve": sum(1 for r in records if r["is_curve"]),
        "usable": sum(1 for r in records if r["usable"]),
    }


def group_into_legs(
    records: list[dict], positions_m: np.ndarray, footprint_along_m: float
) -> list[dict]:
    """Passate: sequenze massimali di scatti non in virata con lo stesso `heading`.

    Ogni passata e' un dict con `frames` (indici in `records`, in ordine temporale) e
    `heading`. Gli scatti in virata non appartengono a nessuna passata; da qui in poi pero'
    non spariscono dal problema, perche' non e' l'appartenenza a una passata a decidere chi
    entra nel mosaico (vedi `select_frames`). La loro odometria resta comunque, ed e' quella
    che trasporta la posizione da una passata alla successiva quando il matching fotografico
    fra passate adiacenti non regge: senza GPS e' l'unica rete di sicurezza rimasta, ed e'
    `utils.poses` a usarla, sommando i delta attraverso la virata.

    Una sequenza corta a meta' di una inversione non e' una passata. La domanda giusta pero'
    non e' quanti scatti contenga -- cinque scatti su un volo da ottocento e cinque su uno da
    venti non vogliono dire la stessa cosa -- ma se COPRA TERRENO NUOVO rispetto alle passate
    vere dello stesso volo. Il metro e' quindi il minore fra due lunghezze, entrambe misurate
    e nessuna delle due tarata a mano:

        l'impronta along-track di un singolo scatto -- il terreno nuovo che una passata
        deve aggiungere perche' valga la pena chiamarla passata;

        una frazione della passata tipica di questo volo, cioe' della mediana delle
        estensioni grezze -- perche' su un rilievo di corridoio NESSUNA passata arriva a
        un'impronta (15-40 m contro 55-66 m sui quattro voli `UgCS-VT*`) e il primo metro da
        solo lasciava il volo senza una sola passata riconosciuta.

    Il minore, non il maggiore: cosi' un volo d'area, dove le passate sono lunghe otto
    impronte, tiene il metro severo che aveva prima -- sul volo di prova la soglia resta
    l'impronta e le 18 passate sono le stesse -- e su un poligono irregolare le passate corte
    ai bordi non vengono giudicate contro quelle lunghe al centro.
    """
    def scarta(leg):
        for i in leg["frames"]:
            records[i]["is_curve"] = True
            records[i]["heading"] = None

    # Il minimo di tre scatti non e' una taratura ma una condizione di forma: due scatti non
    # sono una passata, non stabiliscono una direzione, e se fra i due c'e' un salto -- un
    # volo interrotto e ripreso altrove -- l'estensione da sola li promuove a passata
    # lunghissima. Visto succedere su un volo con un salto di 133 m.
    candidate = []
    for leg in _raw_legs(records):
        if len(leg["frames"]) >= 3:
            candidate.append(leg)
        else:
            scarta(leg)
    if not candidate:
        return []

    estensioni = [leg_extent_m(positions_m, leg["frames"]) for leg in candidate]
    soglia = min(
        footprint_along_m, FRAZIONE_PASSATA_TIPICA * float(np.median(estensioni))
    )

    tenute = []
    for leg, estensione in zip(candidate, estensioni):
        if estensione >= soglia:
            tenute.append(leg)
        else:
            scarta(leg)
    return tenute


def _raw_legs(records: list[dict]) -> list[dict]:
    """Le sequenze prima di qualunque filtro: serve anche a misurare la passata tipica."""
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
    return legs


def leg_extent_m(positions_m: np.ndarray, leg_frames: list[int]) -> float:
    """Distanza fra il primo e l'ultimo scatto di una sequenza, in metri."""
    if len(leg_frames) < 2:
        return 0.0
    return float(np.linalg.norm(positions_m[leg_frames[-1]] - positions_m[leg_frames[0]]))


def select_frames(
    records: list[dict],
    positions_m: np.ndarray,
    legs: list[dict],
    footprint_along_m: float,
    target_overlap: float,
) -> list[int]:
    """I fotogrammi che entrano nel mosaico, in ordine temporale.

    E' la leva contro lo spreco: il volo di prova ha overlap frontale 80%, cioe' quattro
    scatti quasi identici dove ne basterebbero due. Diradare riduce nello stesso colpo le
    coppie da matchare, i parametri da ottimizzare e i frame da comporre.

    La domanda posta a ogni scatto e' "aggiungi copertura?", e si cammina il volo in ordine
    di tempo tenendo uno scatto quando ci si e' allontanati di `passo_minimo` dall'ultimo
    tenuto. Non "appartieni a una passata?": su una greca d'area le due domande hanno la
    stessa risposta, perche' si vira fuori dall'area, ma su un rilievo di corridoio la
    virata sta sopra il terreno da rilevare e la seconda domanda buttava via il volo intero.

    Misurato sulle impronte GPS, contro il NUCLEO dell'area -- il terreno che il volo copre
    con almeno tre scatti, cioe' quello che il piano di volo intendeva rilevare:

                            passate   scatti   nucleo coperto
        Create-Area-Route3    18       387       97,6%     prima: identico
        UgCS-VT3 (624)        82       222       88,2%     prima: 0 passate, niente
        UgCS-VT2 (353)        44       117       89,6%     prima: 0 passate, niente
        UgCS-VT1 (106)         4        29       95,8%     prima: 0 passate, niente
        UgCS-VT3  (36)         3        10       88,2%     prima: 0 passate, niente

    Le ESTREMITA' di ogni passata si tengono sempre, ed e' l'unico punto in cui le passate
    contano ancora: sono gli scatti che coprono i bordi, dove il drone inverte e nessun altro
    scatto arriva. Contano soprattutto sui corridoi, dove una passata e' lunga tre scatti e
    quindi e' quasi tutta bordo -- il solo diradamento sul percorso, senza estremita', copre
    85,9% invece di 88,2% su `VT3` e 83,4% invece di 88,2% sul `VT3` corto.

    Sul volo di prova invece la copertura e' indifferente (97,9% senza, 97,6% con), ma
    l'insieme scelto no: senza estremita' sarebbero 369 fotogrammi DIVERSI, con estremita'
    sono esattamente quelli di prima. Verificato sulla pipeline vera, cioe' sulle posizioni
    dell'odometria e non su quelle GPS: 284 fotogrammi su 835, gli stessi 284 di prima uno
    per uno, quindi il mosaico del volo di prova non cambia di un byte.

    Gli scatti con l'impronta non rettangolare (`usable` falso) non entrano mai, per quanto
    terreno nuovo coprano: non e' spreco, e' che di loro la pipeline non sa dire dove
    guardano.
    """
    passo_minimo = footprint_along_m * max(1.0 - target_overlap, 0.05)
    estremita = {leg["frames"][0] for leg in legs} | {leg["frames"][-1] for leg in legs}

    tenuti: list[int] = []
    ultimo = None
    for i, record in enumerate(records):
        if not record.get("usable", True):
            continue
        if (
            ultimo is None
            or i in estremita
            or float(np.linalg.norm(positions_m[i] - positions_m[ultimo])) >= passo_minimo
        ):
            tenuti.append(i)
            ultimo = i
    return tenuti
