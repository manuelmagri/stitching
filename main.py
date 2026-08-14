"""Punto di ingresso.

    python main.py <volo> [--da N] [--a M] [--overlap-frontale F] [--overlap-laterale L]

Compone un mosaico georeferenziato dalle immagini di un volo drone. Tutto cio' che
caratterizza il volo -- range dei frame, direzione delle passate, scatti in virata,
overlap, risoluzione di lavoro -- viene dedotto dai dati gia' in nostro possesso; le
opzioni servono solo a scavalcare la deduzione quando serve.

Il volo puo' arrivare in due formati di consegna: immagini rettificate piu' telemetria
EXIF/XMP, oppure una ortofoto gia' georeferenziata per ogni scatto. `utils.sources` li
legge entrambi e li riduce alla stessa cosa; da `utils.pairing` in poi nulla distingue i
due casi, e nessun parametro dipende da quale sia.

Il GPS non partecipa allo stitching quando la consegna permette di farne a meno (vedi
`utils`): interviene solo all'ultimo passo, per collocare nel mondo una ricostruzione che
ha gia' scala e orientamento propri. Non avendo contribuito, resta un insieme di
validazione indipendente, e il residuo del fit finale e' una misura onesta di quanto e'
buona la ricostruzione -- 0,70 m RMS sul volo di prova. Sulle ortofoto per scatto quella
separazione non e' possibile, perche' non ci sono velocita' inerziali registrate, e il
programma lo dichiara invece di stampare la stessa frase.

Il mosaico si compone a bande e si scrive man mano nel GeoTIFF: il canvas del volo di
prova e' 1,4 gigapixel e non esiste alcun momento in cui stia tutto in memoria.
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils import (
    compositing,
    features,
    flight,
    footprint,
    frames,
    georef,
    georeference,
    matching,
    mosaic,
    pairing,
    points,
    poses,
    sources,
)
from utils.geodesy import transformers_for


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
OUTPUT_TIFF = OUTPUT_DIR / "mosaic.tif"
OUTPUT_PREVIEW = OUTPUT_DIR / "mosaic_preview.jpg"  # Il GeoTIFF e' troppo grande da guardare
OUTPUT_LAYOUT = OUTPUT_DIR / "mosaic_layout.jpg"

# Quanto si e' disposti a spendere per un giro in piu' di feature e matching quando il
# grafo non regge. Non e' una taratura sulla qualita': e' il confine oltre il quale
# conviene dire che non regge invece di continuare a provare.
TEMPO_MASSIMO_RITENTATIVO_S = 180.0


def _opt_int(raw: str) -> int | None:
    """Intero >= 1, oppure None se il valore e' vuoto o il letterale 'none'."""
    if raw.strip().lower() in ("", "none"):
        return None
    try:
        value = int(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"atteso un intero o 'none', ricevuto {raw!r}")
    if value < 1:
        raise argparse.ArgumentTypeError(f"i frame sono numerati da 1, ricevuto {value}")
    return value


def _opt_overlap(raw: str) -> float | None:
    """Frazione in [0, 1), oppure None se il valore e' vuoto o il letterale 'none'."""
    if raw.strip().lower() in ("", "none"):
        return None
    try:
        value = float(raw)
    except ValueError:
        raise argparse.ArgumentTypeError(f"atteso un numero o 'none', ricevuto {raw!r}")
    if not 0.0 <= value < 1.0:
        raise argparse.ArgumentTypeError(f"overlap fuori dall'intervallo [0, 1): {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Compone un mosaico georeferenziato dalle immagini di un volo drone. "
            "Senza opzioni processa l'intero volo con i valori dedotti dai dati."
        ),
        epilog=(
            "esempi:\n"
            "  python main.py immagini/immagini_senza_distorsione\n"
            "  python main.py immagini/georef\n"
            "  python main.py <volo> --da 2 --a 89\n"
            "  python main.py <volo> --overlap-frontale 0.70 --overlap-laterale none"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "volo",
        type=Path,
        help="cartella con le immagini del volo (.jpg rettificate, oppure ortofoto .tif)",
    )
    parser.add_argument(
        "--da",
        type=_opt_int,
        default=None,
        metavar="N",
        help="primo frame dell'intervallo; se omesso parte dal primo presente",
    )
    parser.add_argument(
        "--a",
        type=_opt_int,
        default=None,
        metavar="M",
        help="ultimo frame dell'intervallo, incluso; se omesso arriva all'ultimo presente",
    )
    parser.add_argument(
        "--overlap-frontale",
        type=_opt_overlap,
        default=None,
        metavar="F",
        help=(
            "overlap frontale a cui diradare lungo la passata; se omesso usa il valore "
            "dedotto dalla soglia laterale"
        ),
    )
    parser.add_argument(
        "--overlap-laterale",
        type=_opt_overlap,
        default=None,
        metavar="L",
        help=(
            "sovrapposizione minima perche' due scatti vengano accoppiati; se omesso parte "
            "dal valore degli appunti e scende solo se il grafo esce spezzato"
        ),
    )
    parser.add_argument(
        "--sorgente",
        choices=("auto", "exif", "ortho"),
        default="auto",
        help=(
            "formato di consegna del volo: 'exif' per immagini rettificate con telemetria "
            "EXIF/XMP, 'ortho' per una ortofoto georeferenziata per scatto. Con 'auto' lo "
            "deduce dall'estensione dei file"
        ),
    )

    args = parser.parse_args(argv)
    if args.da is not None and args.a is not None and args.a < args.da:
        parser.error(f"--a ({args.a}) deve essere >= --da ({args.da})")

    args.laterale_imposta = args.overlap_laterale is not None
    if args.overlap_laterale is None:
        args.overlap_laterale = 0.25  # foglio 4: sotto, due scatti non si toccano abbastanza
    if args.overlap_frontale is None:
        # Il diradamento piu' spinto che lascia comunque legato ogni scatto sia al vicino
        # sia a quello dopo: con bersaglio T i consecutivi si sovrappongono di T e quelli a
        # salto uno di 2T-1, che deve restare sopra la soglia. Cosi' la catena lungo la
        # passata sopravvive a un aggancio fallito.
        args.overlap_frontale = 0.5 * (1.0 + args.overlap_laterale)
    return args


def _fase(titolo: str) -> None:
    print(f"\n== {titolo}", flush=True)


# Fasi
def carica_volo(args: argparse.Namespace):
    """Input, frame locale metrico, passate, diradamento. Ritorna (sorgente, tenuti)."""
    _fase("Caricamento")
    sorgente = sources.load(args.volo, args.da, args.a, args.sorgente)
    records = sorgente.records
    for riga in sorgente.intro:
        print(f"  {riga}")

    _fase("Frame locale (seminato dal GPS)" if sorgente.gps_seeded else "Frame locale (senza GPS)")
    posizioni = sorgente.positions_m
    gsds = sorgente.gsds
    larghezza_m, altezza_m = footprint.footprint_size_m(
        float(np.median(gsds)), sorgente.image_size
    )
    print(
        f"  estensione {np.ptp(posizioni[:, 0]):.0f} x {np.ptp(posizioni[:, 1]):.0f} m | "
        f"GSD {np.median(gsds):.5f} m/px | impronta {larghezza_m:.1f} x {altezza_m:.1f} m"
    )

    _fase("Passate")
    statistiche = flight.mark_curves(records)
    # Una sequenza e' una passata se copre almeno l'impronta di un singolo scatto: sotto,
    # non aggiunge terreno e sono quasi sempre scatti dispersi dentro una inversione.
    passate = flight.group_into_legs(records, posizioni, altezza_m)
    if not passate:
        raise SystemExit(
            "Nessuna passata riconosciuta: il volo non ha una direzione dominante, "
            "oppure il range di frame selezionato cade tutto dentro una virata."
        )
    print(
        f"  rotta dominante {statistiche['axis_deg']:.2f} gradi | "
        f"{statistiche['curve']} scatti in virata | {len(passate)} passate "
        f"da {min(len(l['frames']) for l in passate)} a "
        f"{max(len(l['frames']) for l in passate)} scatti"
    )

    tenuti = sorted(
        i
        for passata in passate
        for i in flight.subsample_leg(
            posizioni, passata["frames"], altezza_m, args.overlap_frontale
        )
    )
    print(
        f"  diradamento a overlap {args.overlap_frontale:.0%}: "
        f"{len(tenuti)} scatti su {statistiche['straight']} non in virata"
    )
    return sorgente, tenuti


def stima_pose(args, sorgente, tenuti):
    """Pose sul canvas: seed dai sensori, raffinamento dalle immagini.

    Le pose si stimano su immagini ridotte: la localizzazione delle feature non migliora
    abbastanza a piena risoluzione da giustificarne il costo. Ritorna anche il lettore,
    perche' serve di nuovo per il piano di fusione.
    """
    records = sorgente.records
    lettore = frames.FrameReader(
        [records[i]["path"] for i in tenuti],
        sorgente.image_size,
        reduce=frames.reduction_for(sorgente.image_size),
    )
    dimensione = lettore.image_size
    fattore = lettore.scale
    gsds_lavoro = sorgente.gsds[tenuti] * fattore
    gsd_canvas = float(np.median(gsds_lavoro))

    _fase("Pose iniziali")
    iniziali = poses.initial_poses(
        sorgente.positions_m[tenuti],
        [records[i] for i in tenuti],
        gsds_lavoro,
        gsd_canvas,
        dimensione,
    )
    print(
        f"  lavoro a {dimensione[0]}x{dimensione[1]} (fattore {fattore:.5f}), "
        f"GSD canvas {gsd_canvas:.5f} m/px"
    )

    _fase("Grafo delle coppie")
    impronte = footprint.quads_from_poses(iniziali, dimensione)
    if args.laterale_imposta:
        # Chi ha scritto una soglia sulla riga di comando vuole quella, non una dedotta.
        soglia = args.overlap_laterale
        coppie = pairing.candidate_pairs(impronte, soglia)
    else:
        coppie, soglia = pairing.pairs_holding_together(impronte, args.overlap_laterale)
    if not coppie:
        raise SystemExit(
            "Nessuna coppia di scatti si sovrappone abbastanza: le impronte non si toccano. "
            "Controlla il range di frame e la quota di volo."
        )
    if soglia < args.overlap_laterale:
        print(
            f"  soglia abbassata da {args.overlap_laterale:.0%} a {soglia:.0%}: sopra, il "
            f"grafo restava spezzato e le passate non avrebbero avuto legami reciproci"
        )
    print(f"  {len(coppie)} coppie sopra il {soglia:.0%}")

    # Il tetto delle feature si alza solo se il grafo esce malato: un volo che regge al
    # primo tentativo non paga un byte in piu' (vedi `features.feature_ladder`).
    scaletta = features.feature_ladder(len(tenuti))
    for tentativo, tetto in enumerate(scaletta):
        inizio = time.time()
        vincoli = _cerca_vincoli(lettore, coppie, iniziali, dimensione, tetto, len(tenuti))
        durata = time.time() - inizio
        if _grafo_sano(len(tenuti), vincoli) or tentativo == len(scaletta) - 1:
            break
        # Il confronto fra descrittori e' quadratico nel tetto, quindi il giro successivo
        # costa circa il quadrato del rapporto fra i due tetti. Il budget di memoria da
        # solo non basta a fermare la scaletta: su un volo con molte coppie il terzo
        # gradino puo' valere un'ora, e va saputo PRIMA di pagarlo.
        prossimo = scaletta[tentativo + 1]
        stima = durata * (prossimo / tetto) ** 2
        if stima > TEMPO_MASSIMO_RITENTATIVO_S:
            print(
                f"  il grafo non regge, ma salire a {prossimo} feature costerebbe circa "
                f"{stima / 60:.0f} minuti: mi fermo a {tetto} e lo dichiaro"
            )
            break
        print(
            f"  il grafo non regge: rialzo il tetto delle feature da {tetto} a "
            f"{prossimo} e riprovo (circa {stima:.0f} s)"
        )
    _diagnostica_grafo(len(tenuti), vincoli, sorgente)

    _fase("Raffinamento globale")
    vincoli_vo = poses.vo_constraints_from_positions(
        list(range(len(tenuti))), sorgente.positions_m[tenuti], gsd_canvas
    )
    raffinate = poses.refine_poses(
        iniziali, vincoli, dimensione, vo_constraints=vincoli_vo, verbose=0
    )
    print(
        f"  residuo fotografico mediano {_residuo(iniziali, vincoli, dimensione):.2f} px "
        f"-> {_residuo(raffinate, vincoli, dimensione):.2f} px"
    )
    _diagnostica_scala(iniziali, raffinate)
    return raffinate, lettore, gsd_canvas


def _diagnostica_scala(iniziali, raffinate, tolleranza: float = 0.10) -> None:
    """Di quanto il raffinamento ha cambiato la scala di ogni fotogramma, rispetto al seed.

    La scala del seed non e' un'ipotesi: viene da una misura -- quota barometrica e focale
    su una consegna DJI, passo del geotransform su una ortofoto -- ed e' buona al percento.
    Un raffinamento che la sposta del trenta per cento su un fotogramma non sta correggendo
    quella misura: sta sfuggendo attraverso un nodo che non ha abbastanza vincoli per
    trattenerlo, e porta con se' anche i vicini.

    E' il rovescio del residuo fotografico, e va letto insieme a quello: il residuo puo'
    crollare proprio PERCHE' le pose si sono deformate fino a soddisfare vincoli sbagliati.
    """
    def scale(M):
        return float(np.hypot(M[0, 0], M[1, 0]))

    deriva = np.array([scale(b) / scale(a) for a, b in zip(iniziali, raffinate)])
    print(f"  scala dei fotogrammi: da {deriva.min():.3f} a {deriva.max():.3f} volte il seed")
    fuori = int(((deriva < 1 - tolleranza) | (deriva > 1 + tolleranza)).sum())
    if fuori:
        print(
            f"  ATTENZIONE: {fuori} fotogrammi hanno cambiato scala di oltre il "
            f"{tolleranza:.0%}. Il GSD del seed e' una misura, non una stima: uno scarto "
            "cosi' e' quasi sempre un nodo con troppo pochi vincoli che scappa, non una "
            "correzione. Le pose seed potrebbero essere migliori di queste."
        )


def _cerca_vincoli(lettore, coppie, iniziali, dimensione, tetto: int, n: int):
    """Feature ORB e vincoli fotografici, a un dato tetto di feature per scatto."""
    _fase(f"Feature ORB (tetto {tetto} per scatto)")
    rilevatore = features.make_detector(tetto)
    archivio = features.DescriptorStore(n)
    lettore.clear()
    for k in tqdm(range(n), desc="  feature"):
        immagine = lettore.get(k)
        if immagine is not None:
            archivio.set(k, *features.detect(rilevatore, immagine))
    print(
        f"  {archivio.counts().sum()} feature in {archivio.nbytes / 1e6:.0f} MB "
        f"(restano in memoria per tutto il volo)"
    )

    _fase("Matching")
    confrontatore = matching.make_matcher()
    vincoli, pochi_inlier, in_disaccordo = [], 0, 0
    for (i, j, _sovrapposizione) in tqdm(coppie, desc="  match"):
        idx_i, idx_j = matching.match_descriptors(
            confrontatore, archivio.descriptors[i], archivio.descriptors[j]
        )
        H, n_inlier = matching.similarity_from_matches(
            archivio.points[i], archivio.points[j], idx_i, idx_j
        )
        if H is None:
            pochi_inlier += 1
            continue
        if not matching.agrees_with_seed(H, iniziali[i], iniziali[j], dimensione):
            in_disaccordo += 1
            continue
        vincoli.append((i, j, H, n_inlier))
    print(
        f"  {len(vincoli)}/{len(coppie)} vincoli validi "
        f"(scartati: {pochi_inlier} con pochi inlier, {in_disaccordo} in disaccordo col seed)"
    )
    if not vincoli:
        raise SystemExit(
            "Nessun vincolo fotografico valido: senza GPS non c'e' nulla che leghi gli "
            "scatti fra loro. Controlla che le immagini siano quelle giuste e a fuoco."
        )
    return vincoli


def _cicli_indipendenti(n: int, vincoli) -> tuple[int, dict]:
    """Quanti anelli chiude il grafo dei vincoli, e il suo riepilogo.

    E' l'unico modo di leggere onestamente il residuo del raffinamento. Un grafo ad albero
    ne ha zero: ogni vincolo si puo' soddisfare esattamente, il residuo crolla vicino a
    zero, e quel numero non dice nulla sulla qualita' della ricostruzione -- dice solo che
    non c'era ridondanza a contraddirla.
    """
    riepilogo = pairing.summarize(n, [(i, j) for (i, j, _H, _k) in vincoli])
    return len(vincoli) - n + riepilogo["components"], riepilogo


def _grafo_sano(n: int, vincoli) -> bool:
    """Se il grafo abbia abbastanza anelli da rendere significativo il raffinamento.

    Un quarto dei nodi e' una soglia larga: un volo a greca con sovrapposizione decente ne
    chiude un paio per nodo, quindi ci passa sopra di un ordine di grandezza, mentre una
    catena o un albero ci cade sotto subito. Serve a distinguere "va bene" da "non regge",
    non a misurare la qualita'.
    """
    cicli, _ = _cicli_indipendenti(n, vincoli)
    return cicli >= n // 4


def _diagnostica_grafo(n: int, vincoli, sorgente) -> None:
    """Che forma ha il grafo dei vincoli SUPERSTITI, e quanto ci si puo' fidare.

    Va guardata dopo il matching e non prima: quello che regge il mosaico non sono le
    coppie tentate ma i vincoli che ne sono usciti, e fra le due cose ci puo' essere un
    terzo di differenza.

    E' anche la stessa misura su cui `_grafo_sano` decide se rialzare il tetto delle
    feature: qui non si riprova piu', si dichiara com'e' andata.
    """
    cicli, riepilogo = _cicli_indipendenti(n, vincoli)
    print(
        f"  {riepilogo['components']} componenti (la maggiore "
        f"{riepilogo['largest_component']}) | grado {riepilogo['degree_min']}/"
        f"{riepilogo['degree_median']:.0f}/{riepilogo['degree_max']} | "
        f"{cicli} cicli indipendenti"
    )
    if cicli < n // 4:
        print(
            "  ATTENZIONE: pochi cicli, il grafo e' quasi un albero. Il residuo del "
            "raffinamento sara' basso perche' non c'e' ridondanza che lo contraddica, non "
            "perche' la ricostruzione sia buona."
        )
    if riepilogo["components"] > 1:
        # Le coppie sono l'unica cosa che lega le passate fra loro quando il GPS non entra
        # nell'ottimizzazione: componenti separate hanno posizioni reciproche indeterminate.
        ancora = (
            "Le componenti restano dove le mette il GPS, che qui semina anche le posizioni."
            if sorgente.gps_seeded
            else "Senza GPS le loro posizioni reciproche sono indeterminate, e nessun peso "
            "lo compensa."
        )
        print(
            f"  ATTENZIONE: i vincoli superstiti spezzano il volo in "
            f"{riepilogo['components']} componenti ({riepilogo['isolated']} scatti "
            f"isolati). {ancora}"
        )


def _residuo(pose, vincoli, dimensione) -> float:
    """Mediana dello scarto fra i due membri di ogni coppia, in pixel di canvas."""
    w, h = dimensione
    angoli = np.array([[0.0, 0.0, 1.0], [w, 0, 1], [w, h, 1], [0, h, 1]])
    scarti = []
    for (i, j, H, _n) in vincoli:
        da_i = (angoli @ pose[i].T)[:, :2]
        proiettati = (angoli @ np.asarray(H).T)[:, :2]
        da_j = (np.column_stack([proiettati, np.ones(4)]) @ pose[j].T)[:, :2]
        scarti.append(np.linalg.norm(da_i - da_j, axis=1).mean())
    return float(np.median(scarti)) if scarti else 0.0


def georeferenzia(pose, dimensione, gsd_canvas, sorgente, tenuti):
    """Colloca la ricostruzione in UTM. E' l'unico punto in cui entra il GPS."""
    records = sorgente.records
    _fase(
        "Georeferenziazione (chiusura sul GPS)"
        if sorgente.gps_seeded
        else "Georeferenziazione (unico uso del GPS)"
    )
    lat = float(np.mean([r["lat"] for r in records]))
    lon = float(np.mean([r["lon"] for r in records]))
    # Il CRS lo dichiara la sorgente, non lo si sceglie qui: e' quello in cui sono gia'
    # espressi `positions_m` e, sulle ortofoto, `frame_to_utm` e la rotta misurata dal
    # pre-pass. Sceglierne un altro dalla media di lat/lon cadrebbe in una zona diversa su
    # un volo a cavallo di un bordo, e siccome il pre-pass la sua zona la prende dal PRIMO
    # scatto, basterebbe un --da/--a a farle divergere. Il mosaico ne uscirebbe comunque
    # giusto -- il fit assorbe la differenza fra due griglie, che su queste distanze e' una
    # similarita' -- ma `grid_north` smetterebbe di essere vero, e la riga qui sotto
    # direbbe "attesa nulla" su parecchi gradi assorbiti.
    utm_crs = sorgente.utm_crs
    to_utm, to_wgs84 = transformers_for(utm_crs)

    obiettivi, origine = georeference.gps_targets_px(
        [records[i] for i in tenuti], to_utm, gsd_canvas
    )
    centri = georeference.pose_centers_px(pose, dimensione)
    attesa = georeference.expected_rotation_deg(lat, lon, sorgente.grid_north)
    A, info = georeference.fit_to_gps(centri, obiettivi, gsd_canvas, attesa)

    print(f"  {utm_crs}")
    if sorgente.grid_north:
        print(
            f"  rotazione assorbita {info['rotation_deg']:+.3f} gradi, attesa nulla "
            f"perche' il frame locale e' gia' in nord griglia"
        )
    else:
        print(
            f"  rotazione assorbita {info['rotation_deg']:+.3f} gradi, di cui "
            f"{info['expected_rotation_deg']:+.3f} di convergenza del meridiano "
            f"(residuo {info['rotation_residual_deg']:+.3f})"
        )
    print(f"  scala {(info['scale'] - 1) * 100:+.2f}%")
    print(
        f"  scarto dal GPS: mediana {info['median_m']:.2f} m, RMS {info['rms_m']:.2f} m, "
        f"massimo {info['max_m']:.2f} m ({info['outliers']} outlier su {info['n']})"
    )
    if sorgente.gps_seeded:
        print(
            "  NB: il GPS ha seminato anche le posizioni, quindi questo scarto NON e' una "
            "validazione indipendente della ricostruzione."
        )
    return georeference.apply_to_poses(A, pose), origine, utm_crs, to_wgs84


def componi(sorgente, tenuti, pose, lettore_pose, gsd_canvas, origine, utm_crs, to_wgs84):
    """Mosaico a piena risoluzione: piano di fusione, composizione a bande, GeoTIFF."""
    records = sorgente.records
    lettore = frames.FrameReader([records[i]["path"] for i in tenuti], sorgente.image_size)
    dimensione = lettore.image_size
    # Unico ponte fra la risoluzione delle pose e quella del mosaico: sposta solo le
    # traslazioni, perche' immagine e canvas cambiano scala insieme.
    ponte = lettore_pose.scale / lettore.scale
    pose_mosaico = poses.rescale_poses(pose, ponte)
    gsd = gsd_canvas / ponte

    traslate, canvas, offset = mosaic.compute_canvas(pose_mosaico, dimensione)
    _fase("Mosaico")
    print(
        f"  scatti a {dimensione[0]}x{dimensione[1]} | canvas {canvas[0]}x{canvas[1]} = "
        f"{canvas[0] * canvas[1] / 1e6:.0f} Mpx | GSD {gsd:.5f} m/px"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    disegno = mosaic.visualize_layout(
        traslate, dimensione, canvas, labels=[records[i]["index"] for i in tenuti]
    )
    cv2.imwrite(str(OUTPUT_LAYOUT), disegno, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"  layout diagnostico: {OUTPUT_LAYOUT}")

    # Guadagni e cuciture hanno bisogno di vedere tutti gli scatti insieme, quindi si
    # pagano sulle stesse immagini ridotte gia' usate per le pose: il piano non dipende
    # dalla risoluzione a cui poi si compone.
    _fase("Piano di fusione")
    inizio = time.time()
    traslate_piano, canvas_piano, _ = mosaic.compute_canvas(pose, lettore_pose.image_size)
    piano = compositing.plan(
        lettore_pose.get,
        traslate_piano,
        lettore_pose.image_size,
        canvas_piano,
        progress=lambda it: tqdm(it, desc="  warp"),
    )
    guadagni = np.asarray(piano.gains).ravel()
    print(
        f"  guadagni da {guadagni.min():.3f} a {guadagni.max():.3f} "
        f"(deviazione standard {guadagni.std():.3f}) | {time.time() - inizio:.0f} s"
    )
    print(
        f"  {piano.kept_whole} oggetti sottratti al taglio: gli scatti non concordano, "
        f"quindi non stanno sul piano, e spezzarli li rovinerebbe"
        if piano.kept_whole
        else "  nessun oggetto da sottrarre al taglio: gli scatti concordano ovunque"
    )

    _fase("Composizione")
    bande = compositing.compose_bands(
        lettore.get,
        traslate,
        dimensione,
        canvas,
        piano,
        progress=lambda it: tqdm(list(it), desc="  bande"),
    )
    # Da quali fotogrammi e' nato questo file. Il mosaico e i suoi ingressi hanno vite
    # separate -- il pre-pass si rifa', i fotogrammi cambiano cartella -- e senza timbro un
    # GeoTIFF vecchio e' indistinguibile da uno appena fatto.
    percorsi = [records[i]["path"] for i in tenuti]
    timbro = georef.input_stamp(
        sorgente.source_dir,
        percorsi,
        f"{records[tenuti[0]]['index']}..{records[tenuti[-1]]['index']}, "
        f"{len(tenuti)} scatti su {len(records)}",
    )

    origine_canvas = georef.canvas_origin_utm(origine, offset, gsd)
    anteprima = georef.write_geotiff(
        OUTPUT_TIFF,
        canvas,
        origine_canvas,
        gsd,
        utm_crs,
        bande,
        preview_max_side=2000,
        stamp=timbro,
    )
    if anteprima is not None:
        cv2.imwrite(str(OUTPUT_PREVIEW), anteprima, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(
        f"  {OUTPUT_TIFF} ({OUTPUT_TIFF.stat().st_size / 1e6:.0f} MB) | "
        f"{lettore.letture} letture per {len(tenuti)} scatti "
        f"({lettore.letture / len(tenuti):.2f} per scatto)"
    )
    print(f"  anteprima: {OUTPUT_PREVIEW}")
    print(
        f"  timbrato con {timbro['STITCHING_FRAMES_COUNT']} fotogrammi da "
        f"{timbro['STITCHING_FRAMES_DIR']} "
        f"(impronta {timbro['STITCHING_FRAMES_DIGEST']}, del "
        f"{timbro['STITCHING_FRAMES_MTIME']})"
    )
    print(
        f"  per verificarlo: python -m utils.georef "
        f"{OUTPUT_TIFF.relative_to(ROOT).as_posix()} {sorgente.source_dir}"
    )

    _fase("Punti sul mosaico")
    riepiloghi = points.remap_folder(
        sorgente,
        tenuti,
        traslate,
        dimensione,
        origine_canvas,
        gsd,
        to_wgs84,
        OUTPUT_DIR,
        OUTPUT_TIFF.name,
    )
    if not riepiloghi:
        print(f"  nessun .csv da rimappare in {sorgente.source_dir}")
    for riga in riepiloghi:
        print(f"  {riga}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    mancanti = sources.missing_inputs(args.volo, args.sorgente)
    if mancanti:
        print("Errore di configurazione:")
        for riga in mancanti:
            print(f"  - {riga}")
        return 1

    avvio = time.time()
    try:
        sorgente, tenuti = carica_volo(args)
        pose, lettore_pose, gsd_canvas = stima_pose(args, sorgente, tenuti)
        pose, origine, utm_crs, to_wgs84 = georeferenzia(
            pose, lettore_pose.image_size, gsd_canvas, sorgente, tenuti
        )
        componi(
            sorgente, tenuti, pose, lettore_pose, gsd_canvas, origine, utm_crs, to_wgs84
        )
    except (ValueError, RuntimeError) as errore:
        print(f"\nErrore: {errore}", file=sys.stderr)
        return 1

    _fase(f"Fine, in {time.time() - avvio:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
