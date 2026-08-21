"""Helper interno: i conti geometrici di una consegna a ortofoto per scatto.

Sta qui tutto cio' che serve per misurare un'ortofoto e ricampionarla nel fotogramma che
la pipeline si aspetta: il rettangolo di ripresa misurato a terra, l'asse along-track e il
suo verso, le affini fra i tre sistemi in gioco -- pixel del fotogramma, UTM, pixel del
file consegnato -- e il warp.

Il PERCHE' di queste operazioni sta nel docstring di `create_ortho_frames`, che le
orchestra: la' c'e' il motivo, qui i conti.

Tre sistemi di coordinate ricorrono ovunque, e vale la pena tenerli distinti leggendo:

    fotogramma   i pixel del file PRODOTTO: rettangolare, a pixel quadrati, -y lungo la rotta
    UTM          metri proiettati, il sistema in cui si misura e in cui la pipeline lavora
    sorgente     i pixel del file CONSEGNATO: nord-up, a pixel anisotropi, meta' nodata
"""
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio


@dataclass(frozen=True)
class Rettangolo:
    """Il fotogramma di ripresa misurato a terra, in UTM."""

    centro: np.ndarray
    asse_lungo: np.ndarray  # unitario, lungo il lato maggiore
    asse_corto: np.ndarray
    lato_lungo: float
    lato_corto: float


# La consegna
def leggi_tags(path: Path) -> dict:
    """I tag GDAL di una ortofoto, senza toccarne i pixel."""
    with rasterio.open(path) as ds:
        return ds.tags()


def _valid_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Pixel effettivamente coperti dall'ortofoto. Il nodata di questi prodotti e' lo zero."""
    return (image_bgr.any(axis=2)).astype(np.uint8)


def _imread_senza_avvisi(path: Path) -> np.ndarray | None:
    """`cv2.imread`, con spenti gli avvisi che libtiff emette sui tag GeoTIFF.

    OpenCV apre il file con la propria libtiff, che i tag geografici non li conosce e ne
    segnala uno per tag e per scatto: 33550 e 33922 sono il geotransform, 34735-34737 le
    chiavi del CRS, 42112-42113 i metadati GDAL. Sono esattamente i tag che `leggi_tags` e
    `leggi_ortofoto` leggono con rasterio, quindi l'avviso non annuncia nessun problema --
    ma sono sette righe per scatto, centosessantuno su un volo da ventitre', e sommergono
    le misure che questo passo stampa.

    Il livello si rimette com'era subito dopo, perche' e' globale al processo: gli avvisi
    di OpenCV che arrivano dal resto della pipeline devono restare visibili.
    """
    logging = cv2.utils.logging
    precedente = logging.getLogLevel()
    logging.setLogLevel(logging.LOG_LEVEL_ERROR)
    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    finally:
        logging.setLogLevel(precedente)


def leggi_pixel(path: Path) -> np.ndarray:
    """I pixel di una ortofoto, letti al momento del bisogno.

    Stanno FUORI da `leggi_ortofoto` di proposito, ed e' l'unica ragione per cui quella
    funzione non li restituisce. Un'ortofoto Altum decompressa occupa 9,6 MB, e il pre-pass
    deve guardare TUTTI gli scatti prima di poter decidere la dimensione del fotogramma:
    tenere i pixel attaccati alla geometria significherebbe tenere in RAM il volo intero --
    sui 61 scatti di `georef3` il processo passava da 68 a 858 MB -- per un array che serve
    una volta sola, nel warp finale.

    La regola del progetto e' la finestra scorrevole di `frames.FrameReader`, che di
    fotogrammi ne tiene due. Qui ne basta uno.

    Il prezzo e' un secondo decode per scatto, misurato in 40 ms: 0,9 s su `georef`, 2,5 s
    su `georef3`, contro i 583 MB che restavano occupati. Cio' che sopravvive alla prima
    passata scende a 275 MB su `georef3`, quasi tutti maschere -- 194 MB -- e quelle non si
    possono lasciar cadere, perche' `dimensione_comune` le rilegge a ogni giro.
    """
    immagine = _imread_senza_avvisi(path)
    if immagine is None:
        raise ValueError(f"Impossibile leggere {path}")
    return immagine


def leggi_ortofoto(path: Path, to_utm, margine: int) -> dict:
    """Tutto cio' che serve per MISURARE una ortofoto consegnata: geometria, maschera, tag.

    I pixel no, quelli li da' `leggi_pixel` uno scatto per volta: la' il perche'. Il
    risultato di questa funzione invece sopravvive per tutto il pre-pass, ed e' per questo
    che conta cosa ci si mette dentro -- la maschera erosa serve davvero a tutti gli scatti
    insieme, perche' `dimensione_comune` la riwarpa a ogni giro.

    L'impronta valida torna gia' EROSA di `margine` pixel, insieme al resto e non a parte:
    ogni misura che tocca il bordo -- il rettangolo di ripresa, la copertura del fotogramma
    -- va fatta su quella, e tenerle attaccate e' il modo di non sbagliarsi.
    """
    with rasterio.open(path) as ds:
        t = ds.transform
        tags = ds.tags()
        size = (ds.width, ds.height)
        if t.b or t.d:
            raise ValueError(f"{path.name}: geotransform con rotazione, non gestito")

    immagine = leggi_pixel(path)

    def px_to_utm(punti_px: np.ndarray) -> np.ndarray:
        lon = t.c + punti_px[:, 0] * t.a + punti_px[:, 1] * t.b
        lat = t.f + punti_px[:, 0] * t.d + punti_px[:, 1] * t.e
        x, y = to_utm.transform(lon, lat)
        return np.column_stack([x, y])

    def utm_to_px(punti_m: np.ndarray, to_wgs84) -> np.ndarray:
        lon, lat = to_wgs84.transform(punti_m[:, 0], punti_m[:, 1])
        return np.column_stack([(np.asarray(lon) - t.c) / t.a, (np.asarray(lat) - t.f) / t.e])

    # Passo del pixel a terra sui due assi. E' qui che si vede l'anisotropia, e serve piu'
    # avanti per scegliere il GSD comune: si misura in metri e non in gradi, altrimenti i
    # due assi non sarebbero confrontabili.
    origine, passo_x, passo_y = px_to_utm(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))

    nucleo = np.ones((2 * margine + 1, 2 * margine + 1), np.uint8)
    return {
        "path": path,
        "size": size,
        "tags": tags,
        # `immagine` non entra nel dict: qui muore, e i pixel si rileggono nel warp finale.
        "mask_erosa": cv2.erode(_valid_mask(immagine) * 255, nucleo),
        "px_to_utm": px_to_utm,
        "utm_to_px": utm_to_px,
        "gsd_xy": (
            float(np.linalg.norm(passo_x - origine)),
            float(np.linalg.norm(passo_y - origine)),
        ),
    }


def indice(nome: str) -> int:
    """Numero di scatto dal nome file, per ordinare e per --da/--a."""
    numeri = re.findall(r"\d+", Path(nome).stem)
    if not numeri:
        raise ValueError(f"nessun numero di scatto nel nome {nome!r}")
    return int(numeri[0])


# Il rettangolo di ripresa, e da che parte guarda
def footprint_rect_utm(mask_erosa: np.ndarray, px_to_utm) -> Rettangolo:
    """Rettangolo di ripresa in UTM.

    Si stima sulla maschera gia' EROSA, cosi' la frangia scura del bordo non entra nella
    misura, e si stima in METRI e non in pixel: nei pixel del file consegnato il fotogramma
    e' un parallelogramma, perche' l'anisotropia manda in 102 gradi un angolo che a terra
    e' retto, e qualunque `minAreaRect` misurata li' sarebbe quella del parallelogramma e
    non del rettangolo.
    """
    contorni, _ = cv2.findContours(mask_erosa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contorni:
        raise ValueError("nessun pixel valido dopo l'erosione del bordo")

    punti_px = max(contorni, key=cv2.contourArea).reshape(-1, 2).astype(np.float64)
    punti_m = px_to_utm(punti_px)

    # minAreaRect lavora in float32 e con coordinate UTM assolute (10^5..10^6 m) perde
    # i decimetri: si centra prima.
    origine = punti_m.mean(axis=0)
    (cx, cy), (larghezza, altezza), angolo = cv2.minAreaRect(
        (punti_m - origine).astype(np.float32)
    )

    theta = math.radians(angolo)
    asse_w = np.array([math.cos(theta), math.sin(theta)])  # direzione del lato `larghezza`
    asse_h = np.array([-math.sin(theta), math.cos(theta)])
    centro = origine + np.array([cx, cy])

    if larghezza >= altezza:
        return Rettangolo(centro, asse_w, asse_h, float(larghezza), float(altezza))
    return Rettangolo(centro, asse_h, asse_w, float(altezza), float(larghezza))


def _passi_locali(centri: np.ndarray, k: int, passo_minimo: float = 0.30) -> list[np.ndarray]:
    """Spostamenti unitari verso il vicino precedente e il successivo, se significativi."""
    passi = []
    for a, b in ((k, k + 1), (k - 1, k)):
        if a < 0 or b >= len(centri):
            continue
        d = centri[b] - centri[a]
        norma = float(np.linalg.norm(d))
        if norma >= passo_minimo:
            passi.append(d / norma)
    return passi


def lato_corto_e_along_track(centri: np.ndarray, rettangoli: list[Rettangolo]) -> bool:
    """Se sia il lato CORTO del rettangolo a guardare lungo la rotta.

    Non si assume: si vota. Il montaggio abituale mette il lato lungo del sensore
    cross-track, ma nulla vieta a una missione di volare con la camera girata di 90 gradi,
    e sbagliare qui ruota ogni fotogramma di un quarto di giro. Il voto usa tutti gli
    scatti del volo, quindi le due o tre virate -- che spingono nel verso opposto, essendo
    quasi perpendicolari alla passata -- restano in minoranza.
    """
    corto = lungo = 0.0
    for k, r in enumerate(rettangoli):
        for d in _passi_locali(centri, k):
            corto += abs(float(d @ r.asse_corto))
            lungo += abs(float(d @ r.asse_lungo))
    return corto >= lungo


def _avanti(centri: np.ndarray, k: int, asse: np.ndarray) -> np.ndarray:
    """L'asse along-track orientato nel verso di volo.

    L'ambiguita' di 180 gradi la scioglie la rotta, ma non quella centrata: sullo scatto
    che apre una passata la differenza all'indietro attraversa la virata e punta altrove.
    Si guardano entrambi i vicini e si crede a quello piu' allineato con l'asse -- dentro
    la passata sono equivalenti, ai suoi estremi solo uno dei due e' la passata.
    """
    passi = _passi_locali(centri, k)
    if not passi:
        return asse
    d = max(passi, key=lambda p: abs(float(p @ asse)))
    return asse if float(d @ asse) >= 0 else -asse


# Le affini fra i tre sistemi
def affine_campionata(image_size, mappa, n: int = 5) -> tuple[np.ndarray, float]:
    """Affine 3x3 che approssima `mappa` sul riquadro `image_size`, e il residuo massimo.

    Le trasformazioni di cui c'e' bisogno qui passano tutte per lon/lat, che con UTM non
    e' in relazione affine: non basta prendere il geotransform, si campiona su una
    griglia di n*n punti -- angoli compresi -- e si stima. Il residuo torna insieme, cosi'
    l'approssimazione si misura invece di darla per buona.

    Minimi quadrati in float64, non un RANSAC: i punti vengono campionati da una
    trasformazione deterministica, non misurati, quindi non c'e' nessun outlier da
    rigettare e ogni cifra decimale conta. `cv2.estimateAffine2D` vorrebbe float32, e su
    coordinate UTM -- che stanno sui milioni di metri, dove float32 ha passo mezzo metro --
    perderebbe piu' di quanto tutta questa geometria valga.
    """
    w, h = image_size
    u, v = np.meshgrid(np.linspace(0, w, n), np.linspace(0, h, n))
    da = np.column_stack([u.ravel(), v.ravel()])
    a = mappa(da)

    design = np.column_stack([da, np.ones(len(da))])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    return np.vstack([coef.T, [0.0, 0.0, 1.0]]), float(np.abs(design @ coef - a).max())


def _tile_to_utm(centro, avanti, gsd: float, image_size: tuple[int, int]) -> np.ndarray:
    """Affine 3x3 pixel del fotogramma -> UTM, nella convenzione immagine della pipeline.

    L'immagine ha x a destra e y in giu', e la pipeline assume che il suo "su" (-y) sia la
    rotta: con imbardata nulla `utils.poses.initial_pose` manda -y a nord e +x a est. Da
    qui le due direzioni: `avanti` per -y, e `avanti` ruotata di 90 gradi in senso orario
    per +x.
    """
    w, h = image_size
    destra = np.array([avanti[1], -avanti[0]])  # rotta ruotata di 90 gradi in orario
    A = np.eye(3, dtype=np.float64)
    A[:2, 0] = gsd * destra
    A[:2, 1] = -gsd * avanti
    A[:2, 2] = centro - gsd * (w / 2.0) * destra + gsd * (h / 2.0) * avanti
    return A


def _mappa_verso_sorgente(A: np.ndarray, utm_to_px, to_wgs84, image_size):
    """Affine 2x3 pixel del fotogramma -> pixel della sorgente, e il suo residuo.

    La catena vera e' fotogramma -> UTM -> lon/lat -> sorgente, e non e' affine, perche'
    UTM e lon/lat non lo sono fra loro. Su sette metri pero' lo e' a meno del nulla.
    """
    def verso_sorgente(punti_px: np.ndarray) -> np.ndarray:
        in_utm = (np.column_stack([punti_px, np.ones(len(punti_px))]) @ A.T)[:, :2]
        return utm_to_px(in_utm, to_wgs84)

    M, residuo = affine_campionata(image_size, verso_sorgente)
    return M[:2], residuo


def geometrie(letti, rettangoli, centri, corto_along, gsd, image_size, to_wgs84):
    """Per ogni scatto: verso di volo, affine verso UTM, affine verso la sorgente."""
    for k, (info, rett) in enumerate(zip(letti, rettangoli)):
        avanti = _avanti(centri, k, rett.asse_corto if corto_along else rett.asse_lungo)
        A = _tile_to_utm(rett.centro, avanti, gsd, image_size)
        M, residuo = _mappa_verso_sorgente(A, info["utm_to_px"], to_wgs84, image_size)
        yield info, avanti, A, M, residuo


# Il ricampionamento
def warp(sorgente: np.ndarray, M: np.ndarray, image_size, interpolazione: int) -> np.ndarray:
    """Ricampiona la sorgente nel fotogramma. `M` va da fotogramma a sorgente, da qui
    WARP_INVERSE_MAP, che e' proprio il verso in cui `warpAffine` sa gia' lavorare."""
    return cv2.warpAffine(
        sorgente, M, image_size,
        flags=interpolazione | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def dimensione_comune(
    letti, rettangoli, centri, corto_along, gsd, to_wgs84, base, copertura=0.9999
) -> tuple[tuple[int, int], list[float]]:
    """La dimensione piu' grande che sta dentro l'impronta valida di OGNI scatto.

    Il primo tentativo viene dal rettangolo circoscritto piu' piccolo del volo, ma
    `minAreaRect` CIRCOSCRIVE: eccede l'impronta la' dove quella non e' esattamente
    rettangolare, e la striscia di nodata che resta non e' il caso raro di qualche scatto
    sfortunato. Misurato su `georef`: al primo tentativo, 1651x1231, sono sotto soglia
    tutti e 23 gli scatti e il peggiore sta al 98,41% -- l'equivalente di una striscia
    spessa una ventina di pixel su un lato. Invece di indovinare un margine si misura -- si
    prova, si guarda quanto manca, si stringe di quel tanto -- e in tre giri si chiude a
    1591x1171, sessanta pixel per asse, al prezzo di qualche warp di maschere.

    Vale la pena insistere: e' questa funzione che rende vero il contratto "fotogramma
    rettangolare senza nodata", e quindi che tiene `footprint`, `features` e `compositing`
    fuori dal discorso.
    """
    image_size = base
    for _ in range(8):
        frazioni = [
            float(warp(info["mask_erosa"], M, image_size, cv2.INTER_NEAREST).mean() / 255.0)
            for info, _avanti, _A, M, _r in geometrie(
                letti, rettangoli, centri, corto_along, gsd, image_size, to_wgs84
            )
        ]
        if min(frazioni) >= copertura:
            return image_size, frazioni
        # Una striscia spessa t lungo un lato vale t/lato della superficie: da quanto manca
        # si risale a quanto stringere, e si toglie il doppio perche' il fotogramma e'
        # centrato e la striscia potrebbe stare da una parte qualunque.
        manca = 2 * int(np.ceil((1.0 - min(frazioni)) * max(image_size))) + 2
        image_size = (image_size[0] - manca, image_size[1] - manca)

    raise RuntimeError(
        "impossibile trovare una dimensione comune priva di nodata: le impronte dei singoli "
        "scatti sono troppo diverse fra loro perche' un solo fotogramma le contenga tutte"
    )
