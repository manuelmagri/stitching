"""Composizione di un mosaico georeferenziato da un volo drone a greca.

Il GPS non partecipa allo stitching. Serve solo all'ultimo passo, per collocare nel mondo
una ricostruzione che ha gia' scala e orientamento propri:

    scala        <- quota barometrica e focale rettificata
    orientamento <- bussola (nord VERO, non nord griglia UTM)
    forma        <- odometria come seed, immagini come misura
    posizione    <- GPS, e solo qui

L'ordine dei moduli e' l'ordine della pipeline:

    dataset       calibrazione, metadati, percorsi. Verifica che le immagini su disco
                  corrispondano alla calibrazione, prima di ogni altra cosa.
    localframe    odometria integrata -> posizioni locali metriche
    flight        scatti in virata e segmentazione in passate, dal solo assetto
    poses         posa iniziale di ogni scatto sul canvas
    footprint     impronta a terra, derivata dalla posa
    pairing       quali coppie vale la pena matchare, e il grafo e' connesso?
    frames        lettura pigra a finestra scorrevole
    features      ORB, con i descrittori che restano per tutto il volo
    matching      corrispondenze e similarita' fra due scatti
    poses         raffinamento globale ai minimi quadrati
    georeference  fit di similarita' sul GPS: l'unico punto in cui entra
    mosaic        geometria del canvas: dove va a finire ogni scatto
    compositing   guadagni, cuciture, fusione multibanda (Brown-Lowe / Burt-Adelson)
    georef        scrittura del GeoTIFF in UTM

Due passate sul disco, a risoluzioni diverse. La prima stima le pose su immagini ridotte,
perche' la localizzazione delle feature non migliora abbastanza a piena risoluzione da
giustificarne il costo. La seconda compone il mosaico a piena risoluzione. Le lega
`poses.rescale_poses`, che va chiamato con `frames.FrameReader.scale` e non con il fattore
di riduzione nominale: la riduzione arrotonda, e i due numeri non coincidono.
"""
