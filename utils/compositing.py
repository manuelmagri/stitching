"""Fusione degli scatti nel mosaico: guadagni, cuciture, multibanda.

E' la successione classica dello stitching, stabilita da Brown e Lowe (Automatic Panoramic
Image Stitching using Invariant Features, IJCV 2007):

    1. compensazione dei guadagni   un fattore moltiplicativo per scatto, scelto perche'
                                    nelle zone in comune due scatti concordino
    2. ricerca delle cuciture       il confine fra due scatti passa dove le immagini gia'
                                    si somigliano, per programmazione dinamica
    2b. oggetti indivisibili        cio' su cui gli scatti non concordano non sta sul piano
                                    di ricampionamento: lo si toglie al taglio e lo si da'
                                    intero a un solo scatto -- vedi `_keep_objects_whole`
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

Il secondo passo invece, nato per le panoramiche, non basta per un volo: vedi
`_nadir_owner`. In una panoramica le immagini condividono il centro di presa, quindi da
quale di esse arrivi un pixel e' quasi indifferente. In un volo a greca ogni punto e'
coperto da una dozzina di scatti che lo guardano da angoli diversi, e prenderlo dalla
periferia di un fotogramma significa prendere la vista piu' obliqua, quella in cui il
rilievo sposta di piu'. Il taglio, lasciato libero, fa esattamente questo.

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

from utils.mosaic import frame_boxes


@dataclass
class CompositingPlan:
    """Guadagni e maschere di cucitura, stimati su immagini rimpicciolite."""

    gains: np.ndarray
    seam_masks: list[np.ndarray]
    kept_whole: int = 0
    """Quanti oggetti sono stati sottratti al taglio da `_keep_objects_whole`."""

    def compensator(self):
        """Un compensatore ricaricato coi guadagni stimati, usabile a qualunque scala.

        Solo `GainCompensator`, cioe' un guadagno per immagine. Le varianti "a blocchi" di
        OpenCV, che stimano una mappa di guadagni variabile dentro ogni scatto, qui non
        sono utilizzabili: `apply` ridimensiona quella mappa sulle dimensioni dell'immagine
        che riceve, e la composizione a bande le passa un RITAGLIO dello scatto, quindi la
        mappa finirebbe stirata sulla porzione sbagliata. Non e' una perdita: le differenze
        spaziali lente -- la vignettatura, il gradiente radiale del 10% misurato su questo
        volo -- sono basse frequenze, ed e' esattamente cio' che la spline sfuma via da se'.
        """
        comp = cv2.detail.GainCompensator()
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


def _nadir_owner(transforms, image_size, canvas_size, boxes) -> np.ndarray:
    """A ogni pixel del canvas lo scatto che lo guarda piu' vicino al proprio nadir.

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

    E' una regola PER PIXEL, e questo e' il suo limite: niente le impedisce di far passare
    un confine in mezzo a un oggetto. Vedi `_keep_objects_whole`, che lo rimedia usando
    proprio questa partizione.
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

    return proprietario


def _nadir_masks(owner: np.ndarray, boxes, margin: int) -> list[np.ndarray]:
    """La partizione di `_nadir_owner` ritagliata per scatto, e dilatata di `margin`.

    `margin` (in pixel della scala ridotta) e' la liberta' che resta al taglio: dilatando
    il territorio di ciascuno, le cuciture possono spostarsi di quel tanto per aggirare un
    edificio o una siepe, senza pero' poter migrare verso i bordi del fotogramma.
    """
    nucleo = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8) if margin > 0 else None
    maschere = []
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        if x1 <= x0 or y1 <= y0:
            maschere.append(np.zeros((1, 1), np.uint8))
            continue
        m = (owner[y0:y1, x0:x1] == i).astype(np.uint8) * 255
        if nucleo is not None:
            m = cv2.dilate(m, nucleo)
        maschere.append(m)
    return maschere


def _disagreement(images, footprints, boxes, canvas_size):
    """Quanto gli scatti che coprono ogni pixel del canvas discordano fra loro.

    Deviazione standard dei livelli di grigio, calcolata SUGLI SCATTI GIA' COMPENSATI: se
    entrasse anche la differenza di esposizione, ogni sovrapposizione risulterebbe in
    disaccordo e la misura non direbbe piu' niente sulla geometria.

    Dove il terreno e' piatto e le pose sono giuste due scatti danno lo stesso pixel, e
    quello che resta e' rumore. Dove qualcosa sta SOPRA il piano su cui la pipeline
    ricampiona, no: un oggetto alto `a` visto a distanza `r` dal nadir si sposta di
    `r * a / H`, quindi due scatti lo mettono in due posti diversi e sui suoi bordi la
    deviazione esplode. Sul volo di prova (quota 8 m, GSD 4,31 mm/px) la mediana sulla
    sovrapposizione e' 6,0 livelli e il massimo 85,5, tutto concentrato sulla cassa.

    Ritorna (disaccordo, copertura), dove `copertura` e' quanti scatti coprono ogni pixel;
    il disaccordo e' nullo dove non ce n'e' almeno due, perche' non c'e' nulla da confrontare.
    """
    canvas_w, canvas_h = canvas_size
    somma = np.zeros((canvas_h, canvas_w), np.float32)
    somma2 = np.zeros((canvas_h, canvas_w), np.float32)
    quanti = np.zeros((canvas_h, canvas_w), np.float32)

    for i, (x0, y0, x1, y1) in enumerate(boxes):
        # Uno scatto illeggibile ha un segnaposto 1x1 al posto del warp (vedi `plan`): non
        # combacia col suo riquadro, e sommarlo qui rovinerebbe la statistica di tutti.
        if x1 <= x0 or y1 <= y0 or footprints[i].shape != (y1 - y0, x1 - x0):
            continue
        grigio = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        valido = (footprints[i] > 0).astype(np.float32)
        grigio *= valido
        fetta = (slice(y0, y1), slice(x0, x1))
        somma[fetta] += grigio
        somma2[fetta] += grigio * grigio
        quanti[fetta] += valido

    n = np.maximum(quanti, 1.0)
    disaccordo = np.sqrt(np.maximum(somma2 / n - (somma / n) ** 2, 0.0))
    disaccordo[quanti < 2] = 0.0
    return disaccordo, quanti.astype(np.int16)


def _keep_objects_whole(masks, footprints, boxes, owner, disagreement, coverage, margin):
    """Toglie al taglio gli oggetti che spezzerebbe, dandoli interi a un solo scatto.

    `_nadir_owner` decide pixel per pixel, e su un oggetto rialzato questo e' il modo
    sbagliato di decidere: il confine fra due territori gli passa in mezzo, i due scatti lo
    proiettano in due posti diversi, e il mosaico incolla la meta' di uno accanto al terreno
    dell'altro. Sul volo di prova la cassa e' finita cosi', tagliata a 89 pixel (38 cm) di
    distanza dalla sua posizione nell'altro scatto.

    Allargare `nadir_margin` non lo risolve -- provato a 0,74, 1,49 e 2,97 m e con
    graphcut: peggiora, perche' il taglio segue la trave dell'oggetto, che e' un gradiente
    forte dove i due scatti concordano di colore. Il taglio non sa che cos'e' un oggetto.
    Quello che si puo' misurare invece si': dove gli scatti discordano c'e' qualcosa che non
    sta sul piano, e la' il taglio non deve passare.

    Tre criteri, nessuno dei quali e' un numero da tarare:

    - **soglia**: mediana + 3 sigma_MAD del disaccordo sulla sovrapposizione. E' il
      criterio di outlier standard, e qui non e' critico: la cassa viene trovata per
      qualunque moltiplicatore fra 1,5 e 6 (a 8 sparisce, sotto 2,5 le macchie iniziano a
      fondersi fra loro).
    - **dimensione**: si protegge solo cio' che e' piu' grande di un disco di raggio
      `margin`, perche' sotto quella taglia il taglio lo aggira gia' da se'. E' il filtro
      che conta davvero: sul volo di prova a 3 sigma le macchie sono 2999, quelle sopra la
      soglia di dimensione 8.
    - **padrone**: chi possiede il CONTORNO A COPERTURA SINGOLA, cioe' l'anello largo
      `margin` attorno alla macchia dove un solo scatto arriva. Quei pixel sono forzati --
      nessun altro puo' fornirli -- quindi se la macchia va a qualcun altro l'oggetto resta
      spezzato fra i due, con certezza. E' l'unica parte del problema che non ha scelta, e
      per questo e' quella che deve decidere.

    Le alternative sono state misurate e sbagliano tutte sulla cassa, che e' contesa fra due
    soli scatti a pari obliquita' (2,93 m contro 3,02 m dal rispettivo nadir):

        chi ha piu' territorio DENTRO la macchia   24:48%  8:39%   -> 24, per 9 punti
        chi e' piu' nadirale sulla macchia         24:161  8:182   -> 24
        chi CONCORDA di piu' con gli altri         24:26,2 8:27,5  -> 24
        chi possiede il contorno forzato           8:75%   24:25%  -> 8, su 2279 px

    I primi tre sono monetine, e la monetina cade dalla parte sbagliata: lo scatto 24 la
    vede di fianco, e il mosaico esce con mezza cassa vista dall'alto e mezza di profilo.
    Il quarto e' netto, e sulle altre sette macchie del volo di prova conferma al 100% la
    stessa scelta degli altri criteri -- non e' un caso particolare cucito addosso.

    Il motivo per cui gli altri criteri non possono funzionare e' che la macchia NON e'
    l'oggetto: e' la parte di oggetto su cui gli scatti discordano, e sulla cassa e' solo la
    meta' destra, perche' a sinistra arriva un solo scatto e non c'e' niente da confrontare.
    Qualunque criterio che guardi solo dentro la macchia sta guardando meta' del problema.

    Quando l'anello non ha pixel a copertura singola non c'e' nulla di forzato, e si ripiega
    sul territorio dentro la macchia. Modifica `masks` sul posto e ritorna quante macchie ha
    assegnato.
    """
    valori = disagreement[coverage >= 2]
    if valori.size == 0:
        return 0
    mediana = float(np.median(valori))
    mad = float(np.median(np.abs(valori - mediana)))
    soglia = mediana + 3.0 * 1.4826 * mad

    # I bordi dell'oggetto sono cio' che discorda; l'interno spesso no, perche' due viste di
    # una superficie uniforme si somigliano comunque. Chiudere e riempire i contorni
    # restituisce l'oggetto come regione piena, che e' quello che va tenuto insieme.
    grezza = cv2.morphologyEx(
        (disagreement > soglia).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    )
    contorni, _ = cv2.findContours(grezza, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    piena = np.zeros_like(grezza)
    cv2.drawContours(piena, contorni, -1, 1, cv2.FILLED)
    quante, etichette, stats, _centroidi = cv2.connectedComponentsWithStats(piena, 8)

    raggio = max(margin, 1)
    minima = np.pi * raggio ** 2
    canvas_h, canvas_w = disagreement.shape
    assegnate = 0
    for k in range(1, quante):
        x, y, bw, bh, area = stats[k]
        if area < minima:
            continue
        macchia = etichette[y : y + bh, x : x + bw] == k

        # L'anello attorno alla macchia, largo quanto la liberta' del taglio, e dentro di
        # esso i pixel che un solo scatto puo' fornire: sono quelli che decidono.
        px0, py0 = max(x - raggio - 1, 0), max(y - raggio - 1, 0)
        px1, py1 = min(x + bw + raggio + 1, canvas_w), min(y + bh + raggio + 1, canvas_h)
        gonfia = np.zeros((py1 - py0, px1 - px0), np.uint8)
        gonfia[y - py0 : y + bh - py0, x - px0 : x + bw - px0] = macchia
        anello = cv2.dilate(gonfia, np.ones((2 * raggio + 1, 2 * raggio + 1), np.uint8))
        forzati = (anello > 0) & (gonfia == 0) & (coverage[py0:py1, px0:px1] == 1)

        voti = owner[py0:py1, px0:px1][forzati]
        voti = voti[voti >= 0]
        if voti.size == 0:
            voti = owner[y : y + bh, x : x + bw][macchia]
            voti = voti[voti >= 0]
        if voti.size == 0:
            continue
        candidati, conteggi = np.unique(voti, return_counts=True)

        padrone, coperto = None, None
        for i in candidati[np.argsort(-conteggi)]:
            bx0, by0, bx1, by1 = boxes[i]
            if footprints[i].shape != (by1 - by0, bx1 - bx0):
                continue
            ix0, iy0 = max(bx0, x), max(by0, y)
            ix1, iy1 = min(bx1, x + bw), min(by1, y + bh)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            impronta = np.zeros((bh, bw), np.uint8)
            impronta[iy0 - y : iy1 - y, ix0 - x : ix1 - x] = footprints[i][
                iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0
            ]
            if int((impronta[macchia] == 0).sum()) < minima:
                padrone, coperto = int(i), impronta > 0
                break
        if padrone is None:
            continue

        # Solo dove il padrone puo' davvero mettere pixel: quel che gli resta fuori e' sotto
        # la soglia di dimensione, e lasciarlo com'e' e' meglio che aprire un buco.
        da_dare = macchia & coperto
        for i, (bx0, by0, bx1, by1) in enumerate(boxes):
            if bx1 <= x or bx0 >= x + bw or by1 <= y or by0 >= y + bh:
                continue
            if masks[i].shape != (by1 - by0, bx1 - bx0):
                continue
            ix0, iy0 = max(bx0, x), max(by0, y)
            ix1, iy1 = min(bx1, x + bw), min(by1, y + bh)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            fetta = da_dare[iy0 - y : iy1 - y, ix0 - x : ix1 - x]
            masks[i][iy0 - by0 : iy1 - by0, ix0 - bx0 : ix1 - bx0][fetta] = (
                255 if i == padrone else 0
            )
        assegnate += 1
    return assegnate


def plan(
    image_loader,
    transforms: list[np.ndarray],
    image_size: tuple[int, int],
    canvas_size: tuple[int, int],
    seam_megapix: float = 0.1,
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

    `nadir_margin` e' di quanti pixel (alla scala ridotta) il taglio puo' allontanarsi dal
    territorio nadirale di ogni scatto; 0 lo blocca li'.

    Il cercatore di cuciture e' quello per programmazione dinamica. Misurando i tre
    disponibili sulla stessa finestra, come varianza del laplaciano del composito:

        nessuno       930,6   il taglio serve: la partizione nadirale da sola taglia dove
                              capita, e la spline impasta cio' che non combacia
        dp          1.383,9
        graphcut    1.393,4   lo 0,7% sopra dp, indistinguibile a occhio

    Graphcut costa molto di piu' e cresce molto piu' in fretta: 29,3 s contro 23,0 con 40
    scatti, ma 244,3 contro 69,9 con 80. E' l'unica fase che non scala col volo, ed e' la
    ragione per cui va pagata su immagini piccole: da qui `seam_megapix`.
    """
    w, h = image_size
    scale = min(1.0, float(np.sqrt(seam_megapix * 1e6 / max(w * h, 1))))
    size_s = (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1))
    canvas_s = (
        max(int(round(canvas_size[0] * scale)), 1),
        max(int(round(canvas_size[1] * scale)), 1),
    )
    M_s = [_scaled(M, scale) for M in transforms]
    boxes = frame_boxes(M_s, size_s, canvas_s)

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

    comp = cv2.detail.GainCompensator()
    comp.feed(angoli, immagini, maschere)
    gains = np.asarray(comp.getMatGains())

    # Le cuciture si cercano sulle immagini GIA' compensate: se due scatti differiscono
    # solo di esposizione, il taglio pagherebbe quel salto come se fosse un disallineamento
    # e sceglierebbe un percorso senza senso.
    immagini = [
        comp.apply(i, angoli[i], immagini[i], maschere[i]) for i in range(len(immagini))
    ]

    # Il taglio parte dal territorio nadirale di ciascuno, non dall'intera impronta:
    # senza questo vincolo migrerebbe verso i bordi dei fotogrammi. I guadagni invece sono
    # gia' stati stimati sull'intera sovrapposizione, che e' piu' dati e stime migliori.
    proprietario = _nadir_owner(M_s, size_s, canvas_s, boxes)
    iniziali = _nadir_masks(proprietario, boxes, nadir_margin)
    iniziali = [cv2.bitwise_and(a, b) for a, b in zip(iniziali, maschere)]

    # Poi gli si tolgono di mezzo gli oggetti che spezzerebbe. Va fatto PRIMA del cercatore
    # e non dopo: una macchia che appartiene a un solo scatto non e' piu' sovrapposizione,
    # quindi il taglio non ha nulla da cercarci ed e' costretto ad aggirarla, invece di
    # sceglierne il percorso e vederselo poi sovrascrivere.
    interi = 0
    if len(transforms) > 1:
        disaccordo, copertura = _disagreement(immagini, maschere, boxes, canvas_s)
        interi = _keep_objects_whole(
            iniziali, maschere, boxes, proprietario, disaccordo, copertura, nadir_margin
        )
        del disaccordo, copertura
        iniziali = [cv2.bitwise_and(a, b) for a, b in zip(iniziali, maschere)]

    finder = cv2.detail.DpSeamFinder("COLOR_GRAD")
    uscita = finder.find(
        [im.astype(np.float32) / 255.0 for im in immagini],
        angoli,
        [cv2.UMat(m) for m in iniziali],
    )
    seam_masks = [m.get() if isinstance(m, cv2.UMat) else np.asarray(m) for m in uscita]

    return CompositingPlan(gains=gains, seam_masks=seam_masks, kept_whole=interi)


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
    boxes = frame_boxes(transforms, image_size, canvas_size)
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
