"""Punto di ingresso.

    python main.py <volo> [--da N] [--a M] [--overlap-frontale F] [--overlap-laterale L]

Compone un mosaico georeferenziato dalle immagini di un volo drone. Tutto cio' che
caratterizza il volo -- range dei frame, direzione delle passate, scatti in virata,
overlap, risoluzione di lavoro -- viene dedotto dai dati gia' in nostro possesso; le
opzioni servono solo a scavalcare la deduzione quando serve.

Il GPS non partecipa allo stitching. La ricostruzione nasce con scala propria (quota
barometrica e focale), orientamento proprio (bussola) e forma propria (odometria come
seed, immagini come misura); il GPS interviene solo all'ultimo passo, per collocarla nel
mondo. Il vantaggio non e' ideologico: non avendo contribuito, il GPS resta un insieme di
validazione indipendente, e il residuo del fit finale e' una misura onesta di quanto e'
buona la ricostruzione. Sul volo di prova quel residuo e' 0,70 m RMS.

Due passate sul disco, a risoluzioni diverse. La prima stima le pose su immagini ridotte,
perche' la localizzazione delle feature non migliora abbastanza a piena risoluzione da
giustificarne il costo. La seconda compone il mosaico a piena risoluzione, a bande, e lo
scrive man mano nel GeoTIFF: il canvas del volo di prova e' 1,4 gigapixel e non esiste
alcun momento in cui stia tutto in memoria.
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
    dataset,
    features,
    flight,
    footprint,
    frames,
    georef,
    georeference,
    localframe,
    matching,
    mosaic,
    pairing,
    poses,
)
from utils.geodesy import make_transformers


# Path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CALIBRATION_FILE = DATA_DIR / "calibration.json"    # Prodotto da preprocessing/create_calibration.py
METADATA_FILE = DATA_DIR / "metadata.json"          # Prodotto da preprocessing/create_metadata.py
TRANSLATIONS_FILE = DATA_DIR / "translations.json"  # Prodotto da preprocessing/create_translations.py
OUTPUT_DIR = ROOT / "output"
OUTPUT_TIFF = OUTPUT_DIR / "mosaic.tif"
OUTPUT_PREVIEW = OUTPUT_DIR / "mosaic_preview.jpg"  # Il GeoTIFF e' troppo grande da guardare
OUTPUT_LAYOUT = OUTPUT_DIR / "mosaic_layout.jpg"


# CLI
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
            "  python main.py <volo> --da 2 --a 89\n"
            "  python main.py <volo> --overlap-frontale 0.70 --overlap-laterale none"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "volo",
        type=Path,
        help="cartella con le immagini rettificate del volo (nomi ..._NNNN_D.JPG)",
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
            "sovrapposizione minima perche' due scatti vengano accoppiati; se omesso usa "
            "il valore degli appunti"
        ),
    )

    args = parser.parse_args(argv)
    if args.da is not None and args.a is not None and args.a < args.da:
        parser.error(f"--a ({args.a}) deve essere >= --da ({args.da})")

    # Deduzione dei valori omessi.
    if args.overlap_laterale is None:
        args.overlap_laterale = 0.25  # foglio 4: sotto, due scatti non si toccano abbastanza
    if args.overlap_frontale is None:
        # Il diradamento piu' spinto che lascia comunque legato ogni scatto sia al vicino
        # sia a quello dopo: con bersaglio T i consecutivi si sovrappongono di T e quelli a
        # salto uno di 2T-1, che deve restare sopra la soglia. Cosi' la catena lungo la
        # passata sopravvive a un aggancio fallito.
        args.overlap_frontale = 0.5 * (1.0 + args.overlap_laterale)
    return args


def _check_inputs(images_dir: Path) -> list[str]:
    missing: list[str] = []
    for path in (CALIBRATION_FILE, METADATA_FILE, TRANSLATIONS_FILE):
        if not path.is_file():
            missing.append(f"file mancante: {path}  (generalo con preprocessing/)")
    if not images_dir.is_dir():
        missing.append(f"cartella mancante: {images_dir}")
    elif not any(p.suffix.lower() == ".jpg" for p in images_dir.iterdir()):
        missing.append(f"nessun .jpg in {images_dir}")
    return missing


def _fase(titolo: str) -> None:
    print(f"\n== {titolo}", flush=True)


# Fasi
def carica_volo(args: argparse.Namespace):
    """Input, frame locale metrico, passate, diradamento.

    Ritorna (calibrazione, records, posizioni, gsds, passate, indici_tenuti).
    """
    _fase("Caricamento")
    calibrazione = dataset.load_calibration(CALIBRATION_FILE)
    records = dataset.load_records(METADATA_FILE)
    records = dataset.select_range(
        records, dataset.list_image_paths(args.volo), args.da, args.a
    )
    # La focale descrive le immagini RETTIFICATE: se i file su disco non sono quelli,
    # ogni misura metrica a valle e' sfasata dello stesso fattore, in silenzio.
    dataset.verify_image_size(calibrazione, records[0]["path"])
    print(
        f"  {len(records)} scatti (frame {records[0]['index']}..{records[-1]['index']}), "
        f"immagini {calibrazione.image_size[0]}x{calibrazione.image_size[1]}, "
        f"focale {calibrazione.focal_px:.2f} px"
    )

    _fase("Frame locale (senza GPS)")
    posizioni = localframe.positions_for(
        records, localframe.integrate_positions(localframe.load_vo_deltas(TRANSLATIONS_FILE))
    )
    gsds = localframe.frame_gsds(records, calibrazione.focal_px)
    larghezza_m, altezza_m = footprint.footprint_size_m(
        float(np.median(gsds)), calibrazione.image_size
    )
    print(
        f"  estensione {np.ptp(posizioni[:, 0]):.0f} x {np.ptp(posizioni[:, 1]):.0f} m | "
        f"GSD {np.median(gsds):.5f} m/px | impronta {larghezza_m:.1f} x {altezza_m:.1f} m"
    )

    _fase("Passate")
    statistiche = flight.mark_curves(records)
    passate = flight.group_into_legs(records)
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
    return calibrazione, records, posizioni, gsds, passate, tenuti


def stima_pose(args, calibrazione, records, posizioni, gsds, tenuti):
    """Pose sul canvas: seed dai sensori, raffinamento dalle immagini.

    Le pose si stimano su immagini ridotte di un quarto: la localizzazione delle feature
    non migliora abbastanza a piena risoluzione da giustificarne il costo. Ritorna anche il
    lettore, perche' serve di nuovo per il piano di fusione.
    """
    lettore = frames.FrameReader(
        [records[i]["path"] for i in tenuti], calibrazione.image_size, reduce=4
    )
    dimensione = lettore.image_size
    fattore = lettore.scale
    gsds_lavoro = gsds[tenuti] * fattore
    gsd_canvas = float(np.median(gsds_lavoro))

    _fase("Pose iniziali")
    iniziali = poses.initial_poses(
        posizioni[tenuti], [records[i] for i in tenuti], gsds_lavoro, gsd_canvas, dimensione
    )
    print(
        f"  lavoro a {dimensione[0]}x{dimensione[1]} (fattore {fattore:.5f}), "
        f"GSD canvas {gsd_canvas:.5f} m/px"
    )

    _fase("Grafo delle coppie")
    impronte = footprint.quads_from_poses(iniziali, dimensione)
    coppie = pairing.candidate_pairs(impronte, args.overlap_laterale)
    riepilogo = pairing.summarize(len(tenuti), coppie)
    print(
        f"  {riepilogo['pairs']} coppie sopra il {args.overlap_laterale:.0%} | "
        f"{riepilogo['components']} componenti (la maggiore {riepilogo['largest_component']}) | "
        f"grado {riepilogo['degree_min']}/{riepilogo['degree_median']:.0f}/{riepilogo['degree_max']}"
    )
    if riepilogo["components"] > 1:
        # Senza GPS le coppie sono l'unica cosa che lega le passate fra loro: componenti
        # separate hanno posizioni reciproche indeterminate, e nessun peso lo compensa.
        print(
            f"  ATTENZIONE: il grafo e' spezzato in {riepilogo['components']} componenti "
            f"({riepilogo['isolated']} scatti isolati). Abbassa --overlap-laterale "
            "oppure alza --overlap-frontale per diradare meno."
        )

    _fase("Feature ORB")
    rilevatore = features.make_detector()
    archivio = features.DescriptorStore(len(tenuti))
    for k in tqdm(range(len(tenuti)), desc="  feature"):
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

    _fase("Raffinamento globale")
    vincoli_vo = poses.vo_constraints_from_positions(
        list(range(len(tenuti))), posizioni[tenuti], gsd_canvas
    )
    raffinate = poses.refine_poses(
        iniziali, vincoli, dimensione, vo_constraints=vincoli_vo, verbose=0
    )
    print(
        f"  residuo fotografico mediano {_residuo(iniziali, vincoli, dimensione):.2f} px "
        f"-> {_residuo(raffinate, vincoli, dimensione):.2f} px"
    )
    return raffinate, lettore, gsd_canvas


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


def georeferenzia(pose, dimensione, gsd_canvas, records, tenuti):
    """Colloca la ricostruzione in UTM. E' l'unico punto in cui entra il GPS."""
    _fase("Georeferenziazione (unico uso del GPS)")
    lat = float(np.mean([r["lat"] for r in records]))
    lon = float(np.mean([r["lon"] for r in records]))
    to_utm, _, utm_crs = make_transformers(lat, lon)

    obiettivi, origine = georeference.gps_targets_px(
        [records[i] for i in tenuti], to_utm, gsd_canvas
    )
    centri = georeference.pose_centers_px(pose, dimensione)
    A, info = georeference.fit_to_gps(centri, obiettivi, gsd_canvas, lat, lon)

    print(f"  {utm_crs}")
    print(
        f"  rotazione assorbita {info['rotation_deg']:+.3f} gradi, di cui "
        f"{-info['convergence_deg']:+.3f} di convergenza del meridiano "
        f"(residuo {info['rotation_residual_deg']:+.3f})"
    )
    print(f"  scala {(info['scale'] - 1) * 100:+.2f}%")
    print(
        f"  scarto dal GPS: mediana {info['median_m']:.2f} m, RMS {info['rms_m']:.2f} m, "
        f"massimo {info['max_m']:.2f} m ({info['outliers']} outlier su {info['n']})"
    )
    return georeference.apply_to_poses(A, pose), origine, utm_crs


def componi(calibrazione, records, tenuti, pose, lettore_pose, gsd_canvas, origine, utm_crs):
    """Mosaico a piena risoluzione: piano di fusione, composizione a bande, GeoTIFF."""
    lettore = frames.FrameReader([records[i]["path"] for i in tenuti], calibrazione.image_size)
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

    _fase("Composizione")
    bande = compositing.compose_bands(
        lettore.get,
        traslate,
        dimensione,
        canvas,
        piano,
        progress=lambda it: tqdm(list(it), desc="  bande"),
    )
    anteprima = georef.write_geotiff(
        OUTPUT_TIFF,
        canvas,
        georef.canvas_origin_utm(origine, offset, gsd),
        gsd,
        utm_crs,
        bande,
        preview_max_side=2000,
    )
    if anteprima is not None:
        cv2.imwrite(str(OUTPUT_PREVIEW), anteprima, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(
        f"  {OUTPUT_TIFF} ({OUTPUT_TIFF.stat().st_size / 1e6:.0f} MB) | "
        f"{lettore.letture} letture per {len(tenuti)} scatti "
        f"({lettore.letture / len(tenuti):.2f} per scatto)"
    )
    print(f"  anteprima: {OUTPUT_PREVIEW}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    mancanti = _check_inputs(args.volo)
    if mancanti:
        print("Errore di configurazione:")
        for riga in mancanti:
            print(f"  - {riga}")
        return 1

    avvio = time.time()
    try:
        calibrazione, records, posizioni, gsds, _passate, tenuti = carica_volo(args)
        pose, lettore_pose, gsd_canvas = stima_pose(
            args, calibrazione, records, posizioni, gsds, tenuti
        )
        pose, origine, utm_crs = georeferenzia(
            pose, lettore_pose.image_size, gsd_canvas, records, tenuti
        )
        componi(
            calibrazione, records, tenuti, pose, lettore_pose, gsd_canvas, origine, utm_crs
        )
    except (ValueError, RuntimeError) as errore:
        print(f"\nErrore: {errore}", file=sys.stderr)
        return 1

    _fase(f"Fine, in {time.time() - avvio:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
