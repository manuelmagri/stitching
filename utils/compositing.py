"""Fusione degli scatti nel mosaico: guadagni, cuciture, multibanda.

E' la pipeline classica dello stitching, nella successione stabilita da Brown e Lowe
(Automatic Panoramic Image Stitching using Invariant Features, IJCV 2007):

    1. compensazione dei guadagni   ogni scatto riceve un fattore moltiplicativo, scelto
                                    perche' nelle zone in comune due scatti concordino
    2. ricerca delle cuciture       il confine fra due scatti passa dove le immagini gia'
                                    si somigliano. Di default per programmazione dinamica;
                                    il taglio di grafo (Kwatra et al., Graphcut Textures,
                                    SIGGRAPH 2003) da' lo stesso risultato ma costa molto
                                    di piu' -- vedi `plan`
    3. fusione multibanda           Burt e Adelson (A Multiresolution Spline With
                                    Application to Image Mosaics, ACM TOG 1983): le basse
                                    frequenze sfumano su una fascia larga, le alte su una
                                    stretta

Il terzo passo e' quello che risolve il problema che rendeva impraticabile la media
semplice. Le pose hanno un residuo di circa 18 pixel a piena risoluzione, quindi mediare
due scatti sdoppierebbe ogni tetto e ogni albero. La spline multibanda invece prende le
alte frequenze, dove sta il dettaglio, da un solo scatto per volta e le fa cambiare
bruscamente lungo la cucitura, mentre fa transitare dolcemente solo le basse frequenze,
dove il disallineamento non si vede.

C'e' pero' un punto in cui questa successione, nata per le panoramiche, non basta per un
volo, ed e' il secondo passo: vedi `_nadir_masks`. In una panoramica le immagini
condividono il centro di presa, quindi da quale di esse arrivi un pixel e' quasi
indifferente. In un volo a greca ogni punto e' coperto da una dozzina di scatti che lo
guardano da angoli diversi, e prenderlo dalla periferia di un fotogramma invece che dal
suo centro significa prendere la vista piu' obliqua, quella in cui il rilievo sposta di
piu'. Il taglio, lasciato libero, fa esattamente questo. Va vincolato.

I primi due passi girano su copie rimpicciolite, il terzo a piena risoluzione: e' la
strategia a due scale di `stitching_detail`, e qui e' obbligata, perche' guadagni e
cuciture hanno bisogno di vedere tutti gli scatti insieme mentre la fusione viene fatta a
bande. I guadagni si trasferiscono da una scala all'altra con `setMatGains`, le maschere
di cucitura ingrandendole.

La fusione a bande equivale a quella in un colpo solo a patto che ogni banda venga
elaborata con un margine piu' largo del raggio di influenza della spline, che con
`num_bands` livelli e' dell'ordine di 2^num_bands pixel; il margine viene poi ritagliato
via. Misurato su un canvas di 19 Mpx con num_bands=5, contro il riferimento a banda unica:

    margine    0   1.455.203 pixel diversi, scarto massimo 34 livelli
    margine   32     590.892                                 8
    margine  128      20.184                                 3
    margine  512         597                                 3

Col default di 512 la differenza e' lo 0,003% dei pixel per al piu' 3 livelli su 255,
sotto il rumore di quantizzazione del JPEG.
"""
from dataclasses import dataclass

import cv2
import numpy as np

# Solo compensatori con un guadagno per immagine (uno scalare, o uno per canale).
#
# OpenCV offre anche le varianti "a blocchi", che stimano una mappa di guadagni variabile
# dentro ogni scatto, ma qui non sono utilizzabili: `apply` ridimensiona quella mappa sulle
# dimensioni dell'immagine che riceve, e la composizione a bande le passa un RITAGLIO dello
# scatto, quindi la mappa finirebbe stirata sulla porzione sbagliata. Non e' una perdita:
# le differenze spaziali lente -- la vignettatura, il gradiente radiale del 10% misurato su
# questo volo -- sono basse frequenze, ed e' esattamente cio' che la spline multibanda
# sfuma via da se'.
_COMPENSATORI = {
    "gain": cv2.detail.GainCompensator,
    "channels": cv2.detail.ChannelsCompensator,
    "no": cv2.detail.NoExposureCompensator,
}


@dataclass
class CompositingPlan:
    """Guadagni e maschere di cucitura, stimati alla scala ridotta `scale`."""

    scale: float
    gains: np.ndarray
    seam_masks: list[np.ndarray]
    seam_boxes: list[tuple[int, int, int, int]]
    compensator_kind: str

    def compensator(self):
        """Un compensatore ricaricato coi guadagni stimati, usabile a qualunque scala."""
        comp = _COMPENSATORI[self.compensator_kind]()
        if self.compensator_kind != "no":
            comp.setMatGains(self.gains)
        return comp


def _scaled(M: np.ndarray, s: float) -> np.ndarray:
    """La stessa similarita' fra spazi rimpiccioliti di s: cambia solo la traslazione."""
    N = M.copy()
    N[0, 2] *= s
    N[1, 2] *= s
    return N


def _warp_into_box(img, M, box):
    """Warp dell'immagine nel riquadro `box` del canvas. Ritorna (warp, maschera valida)."""
    x0, y0, x1, y1 = box
    sposta = np.array([[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]])
    dim = (x1 - x0, y1 - y0)
    warp = cv2.warpPerspective(img, sposta @ M, dim, flags=cv2.INTER_LINEAR)
    piena = np.full(img.shape[:2], 255, np.uint8)
    maschera = cv2.warpPerspective(piena, sposta @ M, dim, flags=cv2.INTER_NEAREST)
    return warp, maschera


def _boxes(transforms, image_size, canvas_size):
    from utils.mosaic import frame_boxes

    return frame_boxes(transforms, image_size, canvas_size)


def _nadir_masks(transforms, image_size, canvas_size, boxes, margin: int):
    """Maschere iniziali: a ogni scatto il territorio in cui e' il piu' nadirale.

    Serve a vincolare la ricerca delle cuciture. I seam finder di OpenCV minimizzano la
    differenza di colore lungo il taglio e nient'altro: in una panoramica va benissimo,
    perche' tutte le immagini condividono il centro di presa, ma in un volo a greca ogni
    punto e' coperto da una dozzina di scatti e il taglio non ha nessuna ragione per
    preferire quello che lo guarda dall'alto. Misurato su questo volo, lasciato libero
    prende il 95% dei pixel oltre meta' raggio del fotogramma, con distanza mediana dal
    centro 0,718 invece di 0,186.

    E' il posto peggiore da cui prenderli: la periferia dell'immagine e' la vista piu'
    obliqua, quindi quella in cui edifici e alberi si spostano di piu' per effetto del
    rilievo. Fondere due scatti che vedono lo stesso edificio inclinato da parti opposte
    produce un fantasma traslucido, e nessuna regolazione della spline lo toglie.

    `margin` (in pixel della scala ridotta) e' la liberta' che resta al taglio: dilatando
    il territorio di ciascuno, le cuciture possono spostarsi di quel tanto per aggirare un
    edificio o una siepe, senza pero' poter migrare verso i bordi del fotogramma.
    """
    w, h = image_size
    canvas_w, canvas_h = canvas_size

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dal_centro = np.sqrt((xx - w / 2.0) ** 2 + (yy - h / 2.0) ** 2).astype(np.float32)
    lontano = float(np.hypot(w, h))

    migliore = np.full((canvas_h, canvas_w), lontano * 2.0, np.float32)
    proprietario = np.full((canvas_h, canvas_w), -1, np.int32)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if x1 <= x0 or y1 <= y0:
            continue
        sposta = np.array([[1.0, 0.0, -float(x0)], [0.0, 1.0, -float(y0)], [0.0, 0.0, 1.0]])
        d = cv2.warpPerspective(
            dal_centro,
            sposta @ transforms[i],
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=lontano * 2.0,
        )
        fetta = (slice(y0, y1), slice(x0, x1))
        vince = d < migliore[fetta]
        migliore[fetta][vince] = d[vince]
        proprietario[fetta][vince] = i

    nucleo = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8) if margin > 0 else None
    maschere = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if x1 <= x0 or y1 <= y0:
            maschere.append(np.zeros((1, 1), np.uint8))
            continue
        m = (proprietario[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        if nucleo is not None:
            m = cv2.dilate(m, nucleo)
        maschere.append(m)
    return maschere


def plan(
    image_loader,
    transforms: list[np.ndarray],
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    seam_megapix: float = 0.1,
    compensator: str = "gain",
    seam_finder: str = "dp",
    nadir_margin: int = 16,
    progress=None,
) -> CompositingPlan:
    """Stima guadagni e cuciture su copie rimpicciolite delle immagini.

    Il rimpicciolimento non si dichiara in fattori ma in `seam_megapix`, l'area a cui
    portare ogni scatto: e' la parametrizzazione di `stitching_detail`, vale per qualunque
    dimensione di partenza, e il valore di default e' il suo.

    `image_loader`, `image_size`, `transforms` e `canvas_size` devono riferirsi tutti alla
    stessa risoluzione, ma NON deve essere quella della composizione finale: il piano e'
    indipendente dalla scala, perche' i guadagni sono numeri e le maschere vengono
    ingrandite al momento dell'uso. Conviene quindi calcolarlo su un lettore gia' ridotto,
    per non rileggere il volo a piena risoluzione solo per rimpicciolirlo.

    `seam_finder`: "dp" (programmazione dinamica) e' il default, "graphcut" (Kwatra) e'
    l'alternativa piu' blasonata, "no" salta il taglio e lascia a ogni scatto il proprio
    territorio nadirale. Il default e' "dp" perche' misurando i tre sulla stessa finestra
    si equivale a graphcut e costa molto meno. Nitidezza del composito, come varianza del
    laplaciano:

        no          930,6    il taglio serve: la partizione nadirale da sola taglia dove
        dp        1.383,9    capita, e la spline impasta cio' che non combacia
        graphcut  1.393,4    lo 0,7% sopra dp, indistinguibile a occhio

    `nadir_margin` e' di quanti pixel (alla scala ridotta) il taglio puo' allontanarsi dal
    territorio nadirale di ogni scatto; 0 lo blocca li', un valore negativo lo lascia libero
    su tutta l'impronta -- che e' il comportamento da panoramica, sbagliato per un volo.

    Attenzione al costo: i cercatori di OpenCV confrontano tutte le coppie di immagini, e
    graphcut cresce molto piu' in fretta di dp -- 29,3 s contro 23,0 con 40 scatti, ma
    244,3 contro 69,9 con 80. E' l'unica fase che non scala col volo, ed e' la ragione per
    cui va pagata su immagini piccole: da qui `seam_megapix`.
    """
    if compensator not in _COMPENSATORI:
        raise ValueError(f"compensatore sconosciuto: {compensator!r}")

    w, h = image_size
    scale = min(1.0, float(np.sqrt(seam_megapix * 1e6 / max(w * h, 1))))
    size_s = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    canvas_s = (
        max(int(round(canvas_size[0] * scale)), 1),
        max(int(round(canvas_size[1] * scale)), 1),
    )
    M_s = [_scaled(M, scale) for M in transforms]
    boxes = _boxes(M_s, size_s, canvas_s)

    indici = range(len(transforms))
    if progress is not None:
        indici = progress(indici)

    immagini, maschere, angoli = [], [], []
    for i in indici:
        img = image_loader(i)
        if img is None:
            immagini.append(np.zeros((1, 1, 3), np.uint8))
            maschere.append(np.zeros((1, 1), np.uint8))
            angoli.append((0, 0))
            continue
        piccola = cv2.resize(img, size_s, interpolation=cv2.INTER_AREA)
        warp, maschera = _warp_into_box(piccola, M_s[i], boxes[i])
        immagini.append(warp)
        maschere.append(maschera)
        angoli.append((boxes[i][0], boxes[i][1]))

    comp = _COMPENSATORI[compensator]()
    comp.feed(angoli, immagini, maschere)
    gains = np.asarray(comp.getMatGains()) if compensator != "no" else np.ones((len(immagini), 1))

    # Le cuciture si cercano sulle immagini GIA' compensate: se due scatti differiscono
    # solo di esposizione, il taglio pagherebbe quel salto come se fosse un disallineamento
    # e sceglierebbe un percorso senza senso.
    if compensator != "no":
        immagini = [
            comp.apply(i, angoli[i], immagini[i], maschere[i]) for i in range(len(immagini))
        ]

    # Il taglio parte dal territorio nadirale di ciascuno, non dall'intera impronta:
    # senza questo vincolo migrerebbe verso i bordi dei fotogrammi. I guadagni invece sono
    # gia' stati stimati sull'intera sovrapposizione, che e' piu' dati e stime migliori.
    iniziali = (
        _nadir_masks(M_s, size_s, canvas_s, boxes, nadir_margin)
        if nadir_margin >= 0
        else maschere
    )
    iniziali = [cv2.bitwise_and(a, b) for a, b in zip(iniziali, maschere)]

    if seam_finder == "no":
        seam_masks = iniziali
    else:
        if seam_finder == "graphcut":
            finder = cv2.detail.GraphCutSeamFinder("COST_COLOR")
        elif seam_finder == "dp":
            finder = cv2.detail.DpSeamFinder("COLOR_GRAD")
        else:
            raise ValueError(f"ricerca cuciture sconosciuta: {seam_finder!r}")

        src = [im.astype(np.float32) / 255.0 for im in immagini]
        uscita = finder.find(src, angoli, [cv2.UMat(m) for m in iniziali])
        seam_masks = [
            m.get() if isinstance(m, cv2.UMat) else np.asarray(m) for m in uscita
        ]

    return CompositingPlan(
        scale=scale,
        gains=gains,
        seam_masks=seam_masks,
        seam_boxes=boxes,
        compensator_kind=compensator,
    )


def _seam_mask_for(plan_: CompositingPlan, i: int, box: tuple[int, int, int, int]) -> np.ndarray:
    """Maschera di cucitura dello scatto i, riportata alle dimensioni di `box`.

    La maschera nasce da un taglio su immagini rimpicciolite, quindi ingrandita ha un
    bordo a gradini. La si dilata prima di ingrandirla, come fa `stitching_detail`: la
    fusione multibanda ammorbidisce comunque il passaggio, e una maschera leggermente
    generosa evita che fra due scatti resti una riga di pixel che non appartiene a nessuno.
    """
    x0, y0, x1, y1 = box
    piccola = plan_.seam_masks[i]
    if piccola.size <= 1:
        return np.zeros((y1 - y0, x1 - x0), np.uint8)
    dilatata = cv2.dilate(piccola, np.ones((3, 3), np.uint8))
    return cv2.resize(dilatata, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def compose_bands(
    image_loader,
    transforms: list[np.ndarray],
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    plan_: CompositingPlan,
    band_height: int = 2048,
    halo: int = 512,
    num_bands: int = 5,
    progress=None,
):
    """Genera (riga_iniziale, banda BGR, maschera) fondendo con la spline multibanda.

    Ogni banda viene elaborata allargata di `halo` righe sopra e sotto, e il margine viene
    ritagliato: e' quello che rende la fusione a bande indistinguibile da una fusione in un
    colpo solo. Il margine deve superare il raggio di influenza della spline, dell'ordine
    di 2^`num_bands` pixel.
    """
    canvas_w, canvas_h = canvas_size
    boxes = _boxes(transforms, image_size, canvas_size)
    comp = plan_.compensator()

    bande = range(0, canvas_h, band_height)
    if progress is not None:
        bande = progress(bande)

    for banda_y0 in bande:
        banda_y1 = min(banda_y0 + band_height, canvas_h)
        roi_y0 = max(banda_y0 - halo, 0)
        roi_y1 = min(banda_y1 + halo, canvas_h)

        blender = cv2.detail.MultiBandBlender(0, num_bands)
        blender.prepare((0, roi_y0, canvas_w, roi_y1 - roi_y0))

        alimentati = 0
        for i, (bx0, by0, bx1, by1) in enumerate(boxes):
            if by1 <= roi_y0 or by0 >= roi_y1 or bx1 <= bx0 or by1 <= by0:
                continue
            ry0, ry1 = max(by0, roi_y0), min(by1, roi_y1)
            if ry1 <= ry0:
                continue

            img = image_loader(i)
            if img is None:
                continue

            warp, valida = _warp_into_box(img, transforms[i], (bx0, ry0, bx1, ry1))
            if plan_.compensator_kind != "no":
                warp = comp.apply(i, (bx0, ry0), warp, valida)

            cucitura = _seam_mask_for(plan_, i, (bx0, by0, bx1, by1))
            cucitura = cucitura[ry0 - by0 : ry1 - by0]
            maschera = cv2.bitwise_and(valida, cucitura)

            blender.feed(warp.astype(np.int16), maschera, (bx0, ry0))
            alimentati += 1

        if alimentati == 0:
            yield banda_y0, np.zeros((banda_y1 - banda_y0, canvas_w, 3), np.uint8), np.zeros(
                (banda_y1 - banda_y0, canvas_w), bool
            )
            continue

        fuso, maschera_fusa = blender.blend(None, None)
        fuso = np.clip(np.asarray(fuso), 0, 255).astype(np.uint8)
        maschera_fusa = np.asarray(maschera_fusa) > 0

        taglio = slice(banda_y0 - roi_y0, banda_y1 - roi_y0)
        yield banda_y0, fuso[taglio], maschera_fusa[taglio]
