"""Composizione di un mosaico georeferenziato da un volo drone a greca.

Il volo puo' arrivare in due formati di consegna, e la prima cosa che fa la pipeline e'
ridurli a uno solo: `sources` dichiara il contratto, due lettori lo riempiono, e da li' in
poi nessun parametro dipende da quale dei due sia. L'unica differenza che sopravvive
riguarda il GPS, e non tocca un calcolo: tocca cio' che si puo' affermare del risultato.

Sugli scatti DJI il GPS non partecipa allo stitching. Serve solo all'ultimo passo, per
collocare nel mondo una ricostruzione che ha gia' scala e orientamento propri:

    scala        <- quota barometrica e focale rettificata
    orientamento <- bussola (nord VERO, non nord griglia UTM)
    forma        <- odometria come seed, immagini come misura
    posizione    <- GPS, e solo qui

Non avendo contribuito, il GPS resta un insieme di validazione indipendente, e il residuo
del fit finale e' una misura onesta della ricostruzione. Sulle ortofoto per scatto non e'
cosi': scala e orientamento vengono dal geotransform, ma velocita' inerziali non ne sono
state registrate, quindi le posizioni le semina il GPS e quel residuo smette di validare
alcunche'. `Source.gps_seeded` porta la distinzione fino a chi stampa.

L'ordine dei moduli e' l'ordine della pipeline:

    sources       il contratto: cosa la pipeline pretende di sapere di un volo
      source_exif   immagini rettificate + telemetria EXIF/XMP, la consegna DJI
        dataset       calibrazione, metadati, percorsi. Verifica che le immagini su
                      disco corrispondano alla calibrazione, prima di ogni altra cosa
        localframe    odometria integrata -> posizioni locali metriche
      source_ortho  una ortofoto per scatto, gia' georeferenziata dal pre-pass
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
    points        i CSV di punti misurati sui singoli scatti, riportati sul mosaico.
                  Accanto al mosaico e non dentro: senza, il GeoTIFF esce identico

I due lettori non condividono una riga: sono due implementazioni indipendenti dello stesso
contratto, e `sources` li importa pigramente. Un volo ortho non carica mai `dataset` e
`localframe`, un volo DJI non carica mai il pre-pass. Vanno letti come due innesti, non
come i rami di una astrazione comune.

`geodesy` non compare nell'elenco perche' non e' un passo: e' il pacchetto di conversioni
-- zona UTM, transformer, GSD, convergenza del meridiano -- che serve ai soli due estremi.

Due passate sul disco, a risoluzioni diverse. La prima stima le pose su immagini ridotte,
perche' la localizzazione delle feature non migliora abbastanza a piena risoluzione da
giustificarne il costo. La seconda compone il mosaico a piena risoluzione. Le lega
`poses.rescale_poses`, che va chiamato con `frames.FrameReader.scale` e non con il fattore
di riduzione nominale: la riduzione arrotonda, e i due numeri non coincidono.
"""
