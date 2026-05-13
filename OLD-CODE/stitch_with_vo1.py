import numpy as np
import cv2
import glob #utilizzata per trovare file e directory che corrispondono a un determinato pattern
import pickle #serve a serializzare e deserializzare oggetti, cioè a trasformare oggetti Python complessi in una rappresentazione binaria che può essere salvata su disco o trasmessa in rete e poi ricostruita successivamente.
#import matplotlib.pyplot as plt
import os
import re
import tqdm #serve per creare barre di progresso per tenere traccia dell'andamento del processo desiderato.
from scipy.spatial.transform import Rotation as R
from utils.detect_feature_in_nxn_quadrants import detect_features_in_nxn_quadrants
from lib.visualization import plotting
from lib.visualization import plot_one
from lib.visualization import plot_separati


class OrtoMosaicSlam():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.start = 0  # Se start e end sono entrambi impostati a 0, esegue i calcoli su tutte le immagini della cartella
        self.end = 135
        self.flag = True
        self.predicted_yaw = []

        # Caricamento delle matrici di calibrazione della fotocamera
        self.cameraMatrix, self.projMatrix = self.load_calib(os.path.join(data_dir, 'data/calibration.pkl'))

        # Caricamento delle immagini
        self.images = self.load_images(os.path.join(data_dir, 'immagini/immagini_drone/immagini_senza_distorsione/*jpg'))
        self.stitched_img = self.images[0]  # Mosaico inizializzato alla prima immagine
        print("Inizializzato il mosaico con la prima immagine!")

        # Caricamento delle pose e delle distanze dai file di dati
        #self.gt_poses, self.distance = self.load_poses(os.path.join(data_dir, 'data/rotations.txt'),os.path.join(data_dir, 'data/translations.txt'))
        self.scales = self.load_scales(os.path.join(data_dir, 'data/scales.txt'))

        # Inizializzazione della ground truth
        #init_gt_pose = self.gt_poses[0] # posa inizile della ground truth
        #print(f"Valore iniziale della ground truth: {init_gt_pose}")
        #self.cur_pose = init_gt_pose #iniziailizzare alla posa iniziale della ground truth
        self.cur_pose = np.eye(4)     #iniziailizzare senza tener conto della ground truth

        # Liste per memorizzare i percorsi
        #self.gt_path = [(float(init_gt_pose[0, 3]), float(init_gt_pose[1, 3]), float(init_gt_pose[2, 3]))]  # Coordinate reali
        self.predicted_path = [(float(self.cur_pose[0, 3]), float(self.cur_pose[1, 3]), float(self.cur_pose[2, 3]))]  # Coordinate stimate

        # Configurazione dell'estrazione delle feature con ORB
        self.orb = cv2.ORB_create(5000)  # Rileva fino a 5000 feature ###controllare non dovrebbe servire

        # Configurazione del matcher FLANN
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

    @staticmethod
    def load_calib(filepath):
        """Carica la calibrazione della fotocamera da un file Pickle."""
        try:
            with open(filepath, 'rb') as f:
                cameraMatrix, projMatrix = pickle.load(f)
        except FileNotFoundError:
            print("Il file .pkl non esiste.")
        except pickle.UnpicklingError:
            print("Errore nel deserializzare il file Pickle.")
        return cameraMatrix, projMatrix

    @staticmethod
    def load_scales(filepath):
        """Carica i valori di scala da un file."""
        scales = []
        with open(filepath, 'r') as file:
            for line in file:
                if line.strip().startswith("Scale"):
                    parts = line.split(":")
                    if len(parts) > 1:
                        scale_value = float(parts[1].strip())
                        scales.append(scale_value)
        return scales

    def load_poses(self, rotation_file, translation_file):
        """Carica le pose della camera estraendo solo le matrici di rotazione e traslazione tra self.start e self.end."""
        poses = []
        distance = []
        with open(rotation_file, 'r') as rot_file, open(translation_file, 'r') as trans_file:
            current_pose = np.eye(4)  # Matrice identità iniziale
            rotation_lines = rot_file.readlines()
            translation_lines = trans_file.readlines()

            # Se start ed end sono entrambi uguali a 0, esegue i calcoli su tutte le righe
            if self.start == 0 and self.end == 0:
                start_index = 0
                end_index = len(rotation_lines) // 5  # Ogni matrice di rotazione occupa 5 righe
            elif self.start != 0 and self.end == 0:
                start_index = self.start
                end_index = len(rotation_lines) // 5  # Fino alla fine
            else:
                start_index = self.start
                end_index = self.end

            i = start_index * 5  # Ogni matrice di rotazione occupa 5 righe
            translation_index = start_index  # Inizia dalla posizione start

            while i < min(end_index * 5, len(rotation_lines)):
                line = rotation_lines[i].strip()
                if line.startswith("Rotation Matrix"):
                    try:
                        # Lettura delle tre righe successive per la matrice di rotazione
                        row1 = rotation_lines[i + 1].strip().replace('[', '').replace(']', '').split()
                        row2 = rotation_lines[i + 2].strip().replace('[', '').replace(']', '').split()
                        row3 = rotation_lines[i + 3].strip().replace('[', '').replace(']', '').split()
                        row1 = np.array([float(x) for x in row1])
                        row2 = np.array([float(x) for x in row2])
                        row3 = np.array([float(x) for x in row3])
                        rotation = np.array([row1, row2, row3])
                        i += 5  # Avanza al prossimo blocco di rotazione
                    except (ValueError, IndexError) as e:
                        print(f"Errore nella lettura della matrice di rotazione alle righe {i + 1}-{i + 3}: {e}")
                        break
                else:
                    i += 1
                    continue

                # Lettura del vettore di traslazione corrispondente
                if translation_index < len(translation_lines):
                    translation_line = translation_lines[translation_index].strip()
                    if translation_line.startswith("Translation Vector"):
                        try:
                            translation = translation_line.split(":")[1].strip().replace('[', '').replace(']', '')
                            translation = np.array([float(x) for x in translation.split()])

                            # Creazione della matrice di trasformazione 4x4
                            transformation = np.eye(4)
                            transformation[:3, :3] = rotation
                            transformation[:3, 3] = translation

                            # Aggiornamento della posizione corrente
                            current_pose = current_pose @ transformation
                            poses.append(current_pose.copy())
                        except ValueError as e:
                            print(f"Errore nella lettura del vettore di traslazione alla riga {translation_index}: {e}")
                            break
                else:
                    print(
                        f"Warning: non è stato trovato un vettore di traslazione per la matrice di rotazione {translation_index}")
                    break
                translation_index += 1

            # Calcolo delle distanze tra le pose successive
            gt_path_3d = np.array(poses)
            for i in range(len(gt_path_3d)):
                if i > 0:
                    distance.append(np.linalg.norm(gt_path_3d[i, :3, 3] - gt_path_3d[i - 1, :3, 3]))
                else:
                    distance.append(1)

            # Salvataggio delle distanze su file
            with open("C:/Users/Asus/Desktop/Progetto_mosaicing/visualOdometry/calibrazione/data/distance.txt",
                      'w') as dist_file:
                for idx, dist in enumerate(distance):
                    dist_file.write(f"Dist {idx + 1}: {dist:.3f} meters\n")

        return poses, distance

    def load_images(self, filepath):
        """
            Carica le immagini da una cartella specificata e le ordina in base al numero nell'filename.

            Args:
                filepath (str): Percorso della cartella contenente le immagini.

            Returns:
                list: Lista di immagini lette e ridimensionate.
        """
        immagini= glob.glob(filepath)# Ottiene tutti i file corrispondenti al percorso specificato

        # Ordinamento,e filtraggio, delle immagini in base al numero presente nel nome del file
        if self.start == 0 and self.end == 0:
            immagini = sorted(immagini,key=lambda x: int(re.search(r'img_(\d+)', x).group(1)))  # considera tutte le immagini
        elif self.start != 0 and self.end == 0:
            immagini = sorted(immagini, key=lambda x: int(re.search(r'img_(\d+)', x).group(1)))[self.start:]
        else:
            immagini = sorted(immagini, key=lambda x: int(re.search(r'img_(\d+)', x).group(1)))[self.start:self.end] #da x compreso a y non compreso

        # Lettura e ridimensionamento delle immagini
        images = []
        for immagine in immagini:
            print("Leggendo immagine : " + str(immagine))
            img = cv2.imread(immagine)  # Carica l'immagine
            height, width = img.shape[:2]  # Ottiene dimensioni dell'immagine
            img = cv2.resize(img, (int(width / 2.5), int(height / 2.5)), interpolation=cv2.INTER_AREA)
            images.append(img)  # Aggiunge l'immagine elaborata alla lista
        return images

    def mixer(self):  # ,i
        """
            Esegue il processo di stitching combinando più immagini e genera il percorso stimato.
        """

        for x in range(1,len(self.images)):# - 1):
            if x < 20:  # salva solo le immagini delle feature e dei match delle prime x immagini
                self.flag = True
            else:
                self.flag = False

            self.stitched_img = self.sticher(x)# Esegue il processo di stitching

        cv2.imwrite(f"output_{self.start}_to_{self.end}.jpg", self.stitched_img)#Salva l'immagine risultante

        #print("gt_path content:", self.gt_path)
        print("pred_path content:", self.predicted_path)
        file_out= f"plot_est_2D_from_{self.start}_to_{self.end}.html"
        file_out=str(file_out)
        plot_one.visualize_path_2d(self.predicted_path,"percorso 2D stimato",file_out=file_out)
        #plot_one.visualize_path_3d(self.predicted_path,"percorso 3D stimato", file_out=os.path.join(self.data_dir, 'data/plot_est_2D.html'))
        #plot_one.visualize_path_2d(self.gt_path, "percorso 2D grund truth", file_out=os.path.join(self.data_dir, 'data/plot_gt_3D.html'))
        #plot_one.visualize_path_3d(self.gt_path, "percorso 3D ground truth", file_out=os.path.join(self.data_dir, 'data/plot_gt_2D.html'))

        pass

    #@staticmethod
    def sticher(self, i):
        """
            Esegue l'unione di immagini utilizzando l'omografia calcolata con i match di feature.

            Args:
                i (int): Indice dell'immagine da incollare allo stitching.

            Returns:
                np.ndarray: Immagine risultante dopo la trasformazione.
        """
        _, _, M = self.get_matches(i)# Ottiene la matrice di trasformazione

        result = self.wrap_images(M, i)# Applica la trasformazione
        result = self.remove_black_borders(result)# Rimuove bordi neri risultanti dalla trasformazione
        return result

    #@staticmethod
    def get_matches(self, i):
        """
        Esegue feature detection e matching tra due immagini utilizzando FLANN con Lowe's Ratio Test,
        dividendo ciascuna immagine in nxn quadranti.
        """
        #immagine del mosaico
        stitch_img = np.array(self.stitched_img, dtype=np.uint8)  # se non è già in uint8

        img1= self.images[i-1]#imagine precente
        img2 = self.images[i] #da incollare allo stitching

        # Controllo di validità delle immagini
        if stitch_img is None or img1 is None or img2 is None:
            raise ValueError("Una delle immagini è vuota!")

        #immagini convertite in scala di grigi pe rendere più facile la rilevazione di feature
        stitch_gray = cv2.cvtColor(stitch_img, cv2.COLOR_BGR2GRAY)
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


        # Percorsi di output per immagini di feature e immagini di feature matching
        feature_dir = "immagini/immagini_drone/feature/"
        match_dir = "immagini/immagini_drone/match/"
        #verifica esistenza dei percorsi delle cartelle d output
        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(match_dir, exist_ok=True)

        #percorso e nome di salvataggio delle immagini
        feature1_path = os.path.join(feature_dir, f"feature_img_{self.start + i - 1}.jpg")
        feature2_path = os.path.join(feature_dir, f"feature_img_{self.start + i}.jpg")
        match_path = os.path.join(match_dir, f"match_{self.start + i - 1}_{self.start + i}.jpg")

        #incrementa il numero di quadranti in cui suddividere l'immagine contenente il mosaico
        #con l'aumentare del numero di immagine incollate
        if i < 5:
            n1 = 5
        elif 5 <= i < 30:
            n1 = 6
        elif 30 <= i < 60:
            n1 = 7
        else:
            n1 = 8
        n2 = 5
        # Estrazione delle feature con rilevamento in regioni suddivise
        keypoints_stitch, descriptors_stitch = detect_features_in_nxn_quadrants(stitch_gray, feature1_path, n1, self.flag, self.start + i - 1)
        keypoints1, descriptors1 = detect_features_in_nxn_quadrants(img1_gray, feature1_path, n2, False,self.start + i -1)
        keypoints2, descriptors2 = detect_features_in_nxn_quadrants(img2_gray, feature2_path, n2, self.flag, self.start + i) # keypoint e descrittori dell'immagine da incollare
        # Controlla la validità dei descrittori
        if descriptors_stitch is None or descriptors1 is None or descriptors2 is None or len(descriptors_stitch) == 0 or len(descriptors1) == 0 or len(descriptors2) == 0:
            print(f"Nessun descrittore trovato tra img_{self.start + i - 1} e img_{self.start + i}.")
            return

        def matches(kp1, des1, kp2, des2):
            """Trova i match tra due set di feature usando FLANN e Lowe's Ratio Test."""
            matches = self.flann.knnMatch(des1, des2, k=2)
            # Applica il test di Lowe: mantiene solo i match in cui il migliore è significativamente migliore del secondo
            good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]#se si diminuisce l'indice trova più match validi, ma avranno meno accuratezza
            if len(good_matches) < 10:
                return [], None, None
            #filtra i keypoint delle due immagini in base ai match trovati (qi e q2 devono avere numero uguale di elementi)
            q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            return good_matches, q1.squeeze(axis=1), q2.squeeze(axis=1)

        good_matches_stitch, q_stitch,q2_1 = matches(keypoints_stitch,descriptors_stitch, keypoints2, descriptors2)
        good_matches, q1, q2_2 = matches(keypoints1, descriptors1, keypoints2, descriptors2)

        # trova l'omografia
        M, mask1 = cv2.findHomography(q_stitch, q2_1, cv2.RANSAC, 3.0)
        #_, mask2 = cv2.findHomography(q1, q2_2, cv2.RANSAC, 3.0)

        if M is not None and mask1 is not None:
            #filtraggio dei match in base alla maschera dell'omografia (util per trovare mach migliori per lo stitching)
            mask1 = mask1.ravel().astype(bool)
            q_stitch = q_stitch[mask1]  # Mantiene solo i punti inliers dalla prima immagine
            q2_1 = q2_1[mask1]  # Mantiene solo i punti inliers dalla seconda immagine

            # Filtrare i match buoni utilizzando la maschera ottenuta da RANSAC
            inlier_matches = [good_matches_stitch[i] for i in range(len(good_matches_stitch)) if mask1[i]]


            # Salva l'immagine con i match
            if self.flag:
                # Disegna i match
                match_image = cv2.drawMatches(stitch_img, keypoints_stitch, img2, keypoints2, inlier_matches, None,
                                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(match_path, match_image)
                print(f"Immagine dei match salvata in {match_path}")

            print(f"In match_{self.start + i - 1}_{self.start + i}.jpg sono stati trovati {len(inlier_matches)} match! ")
            # print(f"Differenza tra good_matches e inlier_matches: {len(good_matches) - len(inlier_matches)}")
        '''
        if mask2 is not None:
            mask2 = mask2.ravel().astype(bool)
            q1 = q1[mask2]
            q2_2 = q2_2[mask2]
            self.get_mvt(q1, q2_2, i)
        '''
        self.get_mvt(q1, q2_2, i) #inizio calcolo del percorso (visual odometry)

        return q_stitch, q2_1, M

    def wrap_images(self, H, i):
        """
            Effettua la trasformazione prospettica per combinare due immagini utilizzando l'omografia calcolata.

            Args:
                H (np.ndarray): Matrice di omografia.
                i (int): Indice dell'immagine corrente.

            Returns:
                np.ndarray: Immagine risultante dopo la trasformazione.
        """
        # Recupera l'immagine corrente e l'immagine già unita
        image1= self.images[i]
        image2= self.stitched_img

        # Ottiene le dimensioni delle immagini
        rows1, cols1 = image1.shape[:2]
        rows2, cols2 = image2.shape[:2]

        # Definisce i quattro angoli dell'immagine corrente
        list_of_points_1 = np.float32(
            [[0, 0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)

        # Definisce i quattro angoli dell'immagine già unita
        temp_points = np.float32(
            [[0, 0], [0, rows2], [cols2, rows2], [cols2, 0]]).reshape(-1, 1, 2)

        # Applica la trasformazione prospettica agli angoli dell'immagine unita
        list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

        # Combina i punti dell'immagine corrente e quelli trasformati dell'immagine unita
        list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

        # Determina i limiti della nuova immagine trasformata
        [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

        # Calcola la traslazione necessaria per mantenere tutti i punti in coordinate positive
        translation_dist = [-x_min, -y_min]

        # Crea la matrice di traslazione per correggere lo spostamento dell'immagine
        H_translation = np.array([[1, 0, translation_dist[0]], [
            0, 1, translation_dist[1]], [0, 0, 1]])

        # Applica la trasformazione prospettica all'immagine unita
        output_img = cv2.warpPerspective(
            image2, H_translation.dot(H), (x_max - x_min, y_max - y_min), flags=cv2.INTER_LANCZOS4)

        # Crea una maschera binaria dell'immagine corrente
        mask = np.ones(image1.shape[:2], dtype=np.float32)  # Maschera uniforme su tutta l'immagine

        # Espandi la maschera per immagini RGB (se necessario)
        if image1.ndim == 3 and image1.shape[2] == 3:  # Verifica se è un'immagine RGB
            mask = np.dstack([mask] * 3)

        # Effettua il blending tra l'immagine corrente e l'immagine trasformata
        blended_region = image1 * mask + output_img[translation_dist[1]:rows1 + translation_dist[1],
                                         translation_dist[0]:cols1 + translation_dist[0]] * (1 - mask)

        # Inserisce l'area fusa nell'immagine finale
        output_img[translation_dist[1]:rows1 + translation_dist[1],
        translation_dist[0]:cols1 + translation_dist[0]] = blended_region

        return output_img

    def remove_black_borders(self, image):
        """
        Rimuove bordi neri da un'immagine.

        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        x, y, w, h = cv2.boundingRect(thresh)
        return image[y:y + h, x:x + w]


    def get_mvt(self,q1,q2, i):
        # una volta trovati i keypoints delle due immagini e i punti che matchano
        # si calocola lo spostamento della fotocamera all'interno dei due frame
        # con l'utilizzo della matrice intrinseca della camera

        Essential, mask = cv2.findEssentialMat(q1, q2, self.cameraMatrix, method=cv2.RANSAC, prob=0.999, threshold=0.5) #La matrice essenziale codifica le informazioni sulla rotazione e traslazione tra due immagini scattate dalla stessa fotocamera
        #r, t = self.decomp_essential_mat2(Essential, q1, q2, i) #scompone la matrice essenziale in possibili matrici di rotazione R e vettori di traslazione t. Poiché ci sono 4 possibili soluzioni (2 rotazioni e 2 traslazioni), la funzione seleziona la soluzone corretta in base ai punti triangolati

        _, r, t, mask_pose = cv2.recoverPose(Essential, q1, q2, self.cameraMatrix)
        t = t * self.scales[i] #moltiplica il vettore di traslazione per la scala per avere una maggiore precisione sulla reale distanza percorsa dal drone tra i frame

        return self.current_pose(r, t, i)

    def transf(self, r, t): #costruisce una matrice di trasformazione omogenea dalla matrice di rotazione R e dal vettore di traslazione t.
        T = np.eye(4, dtype=np.float64) # Inizializza una matrice identità 4x4
        T[:3, :3] = r  # Assegna i primi 3x3 valori a r (rotazione)
        T[:3, 3] = t.ravel()   # Assegna la quarta colonna ai valori di t (traslazione)

        #La matrice risultante permette di comprendere come la fotocamera (o il veicolo) si è spostata tra due istanti consecutivi.
        return T       # Restituisce la matrice omogenea 4x4

    def current_pose(self, r, t, i):
        """
                Calcola e aggiorna la posa attuale del sistema utilizzando la rotazione e la traslazione fornite.

                Args:
                    r (np.ndarray): Matrice di rotazione.
                    t (np.ndarray): Vettore di traslazione.
                    i (int): Indice del frame corrente.
        """
        # Converte la matrice di rotazione in angoli di Eulero
        rot = R.from_matrix(r)
        _, _, yaw = rot.as_euler('xyz', degrees=True)
        # Memorizza l'angolo di imbardata (yaw) predetto
        self.predicted_yaw.append(float(yaw))

        #transf = self.transf(r, np.squeeze(t))
        # Calcola la trasformazione tra la posa attuale e la nuova posizione
        transf = self.transf(r, t)

        #gt_pose= self.gt_poses[i] #estraggo la posizione attuale della ground truth

        # Aggiorna la posa attuale moltiplicando la trasformazione calcolata
        self.cur_pose = self.cur_pose @ transf

        #stampa le posizioni attuali della ground truth e del percorso stimato(visual odometry)
        #print("\nGround truth pose:\n" + str(gt_pose))
        #print("\n Current pose:\n" + str(self.cur_pose))
        # Stampa la posa attuale in coordinate 2D (x, y)
        print("The current 2D pose used x, y: \n" +
              str(self.cur_pose[0, 3]) + "   " +  # Coordinate X
              str(self.cur_pose[1, 3]))  # Coordinate Y
        # Memorizzare le coordinate x, y e z sia per il percorso di ground truth sia per quello stimato
        #self.gt_path.append((float(gt_pose[0, 3]), float(gt_pose[1, 3]), float(gt_pose[2, 3])))  # Pose di ground truth
        self.predicted_path.append((float(self.cur_pose[0, 3]), float(self.cur_pose[1, 3]), float(self.cur_pose[2, 3])))  # Pose stimata
        # Aggiunge le coordinate (x, y, z) della posa di ground truth e della posa stimata
        # rispettivamente a gt_path e estimated_path.
        # Questi percorsi verranno usati per la visualizzazione finale.

    def decomp_essential_mat(self, E, q1, q2, i):
        '''
        cv2.decomposeEssentialMat per decomporre la matrice essenziale E in due possibili
        matrici di rotazione R1 e R2 e un vettore di traslazione t.
        Queste decomposizioni rappresentano le quattro possibili soluzioni per la
        trasformazione rigida tra i due fotogrammi.

        '''

        R1, R2, t = cv2.decomposeEssentialMat(E) #R1 e R2 matrici di rotazione, t array di traslazione
        T1 = self.transf(R1, np.ndarray.flatten(t)) # T* matrice di trasformazione
        T2 = self.transf(R2, np.ndarray.flatten(t))
        T3 = self.transf(R1, np.ndarray.flatten(-t))
        T4 = self.transf(R2, np.ndarray.flatten(-t))
        transformations = [T1, T2, T3, T4]
        '''
        Qui vengono costruite le quattro possibili matrici di trasformazione 4x4, combinando
        le due rotazioni R1 e R2 con il vettore di traslazione t e il suo opposto −t. 
        Ogni trasformazione T descrive una possibile trasformazione tra i due fotogrammi.
        '''

        # Questa operazione aggiunge una colonna di zeri alla matrice intrinseca K,
        # portandola da una dimensione 3x3 a una dimensione 3x4, necessaria per la
        # triangolazione dei punti nello spazio omogeneo.
        K = np.concatenate((self.cameraMatrix, np.zeros((3, 1))), axis=1)

        # Lista delle matrici di proiezione, ottenuta moltiplicando la matrice intrinseca
        # estesa con le varie trasformazioni
        projections = [K @ T1, K @ T2, K @ T3, K @ T4]

        #np.set_printoptions(suppress=True)

        # print ("\nTransform 1\n" +  str(T1))
        # print ("\nTransform 2\n" +  str(T2))
        # print ("\nTransform 3\n" +  str(T3))
        # print ("\nTransform 4\n" +  str(T4))

        def calc_point_spread(Q1, Q2):
            # Calcola la distanza tra i punti in Q1 e Q2
            distances_Q1 = np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)
            distances_Q2 = np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            return np.mean(distances_Q1), np.mean(distances_Q2)  # Media delle distanze

        def reprojection_error(Q1, Q2, K, T):
            # Aggiungi la componente omogenea a Q1 e Q2
            Q1_hom = np.vstack((Q1, np.ones((1, Q1.shape[1]))))  # 4xN
            Q2_hom = np.vstack((Q2, np.ones((1, Q2.shape[1]))))  # 4xN

            # Trasforma i punti con T
            transformed_Q1 = T @ Q1_hom  # 4xN
            transformed_Q2 = T @ Q2_hom  # 4xN

            # Converti da coordinate omogenee a cartesiane
            transformed_Q1 /= transformed_Q1[3, :]  # Divide ogni riga per la quarta riga
            transformed_Q2 /= transformed_Q2[3, :]  # Divide ogni riga per la quarta riga

            # Proietta i punti con K
            projected_Q1 = K @ transformed_Q1[:3, :]  # Proietta i primi 3 componenti
            projected_Q2 = K @ transformed_Q2[:3, :]  # Proietta i primi 3 componenti

            # Normalizza per ottenere le coordinate pixel
            projected_Q1 /= projected_Q1[2, :]
            projected_Q2 /= projected_Q2[2, :]

            # Calcola l'errore di riproiezione
            error_Q1 = np.linalg.norm(projected_Q1[:2, :].T - q1.squeeze(), axis=1)
            error_Q2 = np.linalg.norm(projected_Q2[:2, :].T - q2.squeeze(), axis=1)

            return np.mean(error_Q1 + error_Q2)


        # lista usata per memorizzare il numero di punti triangolati con coordinate z
        # positive (cioè punti validi che si trovano davanti alla fotocamera) per ciascuna
        # delle quattro proiezioni.
        positives = []
        for P, T in zip(projections, transformations):
            #per ogni coppia di matrice di proiezione P e matrice di trasformazione T,
            # vengono triangolati i punti chiave q1 e q2 tra i due fotogrammi
            ##print("Dimensioni di self.projMatrix:", self.projMatrix.shape)
            ##print("Dimensioni di P:", P.shape)
            #print("Numero di punti in q1:", q1.shape[1])
            #print("Numero di punti in q2:", q2.shape[1])

            dynamic_proj_matrix = self.cameraMatrix @ T[:3, :] #####

            hom_Q1 = cv2.triangulatePoints(dynamic_proj_matrix, P, q1.T, q2.T)
            #hom_Q1 = cv2.triangulatePoints(self.projMatrix, P, q1, q2)#q1.T q2.T #punti 3D triangolati nello spazio omogeneo per la prima fotocamera
            #hom_Q1 = cv2.triangulatePoints(dynamic_proj_matrix, P, q1, q2)
            hom_Q2 = T @ hom_Q1 #punti 3D proiettati nella seconda fotocamera utilizzando la matrice di trasformazione T
            #la tringolazione è un metodo per scoprire la distanza di un oggetto(o punto)
            # dalla fotocamera, calcolandola tramite le coordinate dei punti 2D di due frame

            '''
            Usare la stessa matrice di proiezione (self.projMatrix) per tutte le immagini è 
            corretto e utile perché riflette la costanza dei parametri intrinseci della 
            fotocamera. Cambiano solo le trasformazioni estrinseche (rotazione e traslazione
            tra i fotogrammi) che vengono applicate per triangolare i punti 3D tra immagini 
            consecutive. Questo approccio permette di stimare le pose del veicolo/camera 
            in modo efficiente anche su un gran numero di immagini.
            '''

            '''
            Qui, i punti triangolati in coordinate omogenee vengono de-omogeneizzati 
            dividendo le prime tre coordinate (x, y, z) dalla quarta coordinata omogenea. 
            Questo passo restituisce i punti 3D in coordinate cartesiane.
            '''
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

            #In questa riga si contano i punti triangolati che hanno una componente z
            # positiva sia per la prima fotocamera Q1 che per la seconda Q2. Questo passo
            # serve a determinare quale delle quattro trasformazioni produce il maggior numero di punti validi.
            total_sum = sum(Q2[2, :] > 0) + sum(Q1[2, :] > 0)

            #Qui viene calcolata la scala relativa tra i punti triangolati Q1 e Q2.
            #Questa misura confronta la distanza tra i punti consecutivi in Q1 e Q2 per valutare la coerenza tra le due proiezioni
            '''
            norm_diff_Q1 = np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1)
            norm_diff_Q2 = np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1)
            print(f"Valore di norm_diff_Q2: {norm_diff_Q2}")
            valid_indices = norm_diff_Q2 > 1e-6
            if np.any(valid_indices):
                relative_scale = np.mean(norm_diff_Q1[valid_indices] / norm_diff_Q2[valid_indices])
            else:
                relative_scale = 0
            '''
            spread_Q1, spread_Q2 = calc_point_spread(Q1, Q2)

            # Calcola l'errore di riproiezione
            #reprojection_err = reprojection_error(Q1, Q2, self.cameraMatrix, T)

            # Aggiungi il peso dell'errore di riproiezione
            weight = total_sum / (spread_Q1 + spread_Q2) #- reprojection_err
            positives.append(weight) #+ relative_scale) #Il numero totale di punti validi con coordinate z positive viene sommato alla scala relativa calcolata, e il risultato viene aggiunto alla lista positives. Questa somma viene usata per valutare la plausibilità di ciascuna trasformazione.


        print(f"Valori di positivies: {np.int64(positives)}")
        max = np.argmax(positives) #Qui si seleziona l'indice della trasformazione con il valore massimo nella lista positives, cioè quella che produce il maggior numero di punti triangolati validi e coerenti.
        print(f"il valore massimo di positivi si trova in posizione: {max}")
        if (max == 0):
            # print(t)
            return R1, np.ndarray.flatten(t) #* self.scales[i] se voglio utilizzare le scale
        elif (max == 1):
            # print(t)
            return R2, np.ndarray.flatten(t) #* self.scales[i]
        elif (max == 2):
            # print(t)
            return R1, np.ndarray.flatten(-t) #* self.scales[i]
        elif (max == 3):
            # print(t)
            return R2, np.ndarray.flatten(-t) #* self.scales[i]


    def decomp_essential_mat2(self, E, q1, q2, i):
        R1, R2, t = cv2.decomposeEssentialMat(E)
        candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        scores = []
        for R, t in candidates:
            T = self.transf(R, t)
            P = np.matmul(np.concatenate((self.cameraMatrix, np.zeros((3, 1))), axis=1), T)
            dynamic_proj_matrix = self.cameraMatrix @ T[:3, :]
            #hom_Q1 = cv2.triangulatePoints(self.projMatrix, P, q1.T, q2.T)
            hom_Q1 = cv2.triangulatePoints(dynamic_proj_matrix, P, q1.T, q2.T)
            hom_Q2 = T @ hom_Q1
            Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
            Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
            score = sum(Q1[2, :] > 0) + sum(Q2[2, :] > 0)
            scores.append(score)
        R, t = candidates[np.argmax(scores)]
        return R, t


def main():
    data_dir = "Stitching/" #cartella dove è contenuto l'ambiente di lavoro
    #crea un oggetto OrtoMosaicSlam e fa partire il metodo di start chimando il metodo "mixer"
    ort = OrtoMosaicSlam(data_dir)
    ort.mixer()

if __name__ =='__main__':
    main()