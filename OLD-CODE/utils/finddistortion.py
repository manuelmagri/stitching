import numpy as np
import cv2 as cv
import glob #utilizzata per trovare file e directory che corrispondono a un determinato pattern
import pickle #serve a serializzare e deserializzare oggetti, cioè a trasformare oggetti Python complessi in una rappresentazione binaria che può essere salvata su disco o trasmessa in rete e poi ricostruita successivamente.

#############ANGOLI SCACCHIERA########################

chessboardSize = (6,9)  # dimensione in colonne x righe della scacchiera 9x6
frameSize = (5280,3956)  #dimensione dell'immagine della fotocamera
#frameSize = (1872,4056)  #dimensione dell'immagine della fotocamera
#frameSize = (640,480)  #dimensione dell'immagine della fotocamera


# termination criteria                    #############
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 0.001) #aumentare il numero di iterazioni se si vuole una maggiore precisione nel rilevamento degli angoli, ma con maggior tempo di esecuzione
#Significa che l'algoritmo itererà fino a un numero massimo di iterazioni (qui, 30 iterazioni), se non ha già raggiunto la condizione di precisione(0.001.


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32) # matrice (56,3) con terza colonna (che corrisponde alla coordinata z tutta uguale a 0
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2) # prende solo le prime due colonne, quindi evitando la coordinata z
#.T crea la trasposta, in quanto opencv e numpy hanno una diversa implementazione delle coordinate
# reshape(-1,2) mette in ordine la matrice, creando una matrice di 2 colonne(2) e adattando il numero di righe all 2 colonne(-1)
#ogni riga della matrice corrisponde alle coordinate (x,y) di un punto

size_of_chessboard_squares_mm = 25 #diemensione dei quadrati della scacchiera in millimetri
objp = objp * size_of_chessboard_squares_mm

# array per immagazzinare punti dell'oggetto e quelli dell'immagine.
objpoints = [] # punti 3d nello spazio reale
imgpoints = [] # punti 2d dell'immagine piana

immagini = glob.glob('C:/Users/Asus/Desktop/Progetto mosaicing/visualOdometry/immagini/immagini_scacchiera_drone/*.jpg') #crea un array con le immagini presenti nella cartella

for immagine in immagini:

    img = cv.imread(immagine) #legge l'immagine
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) #converte l'immagine in scala di colori bgr ad una in scala di grigi

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)
    #ret=true allora sono stati rilevati degli angoli
    #corners = array di cordinate dei punti degli angoli

    if ret == True:
        objpoints.append(objp) #inserisce nell'array i punti oggetto
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria) # migliora i punti degli angoli con una finestra 22x22, iterando per i termini di criteria
        imgpoints.append(corners2) # inserisce nell'array i punti immagine migliorati

        # disegna e mostra gli angoli trovati
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        #cv.imshow('Immagine', img) #############
        #cv.waitKey(1000) #####################

cv.destroyAllWindows()

###########CALIBRAZIONE###############

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Conversione dei vettori di rotazione in matrici
rotation_matrices = []
for rvec in rvecs:
    R, _ = cv.Rodrigues(rvec)
    rotation_matrices.append(R)

# Ottenimento delle matrici estrinseche e di proriezione
projection_matrices= []
extrinsic_matrices = []
for R, tvec in zip(rotation_matrices, tvecs):
    extrinsic_matrix = np.hstack((R, tvec))
    extrinsic_matrices.append(extrinsic_matrix)

    P = cameraMatrix @ extrinsic_matrix  # @ è l'operatore di prodotto matrice
    projection_matrices.append(P)

i=0
for p in projection_matrices:
    print("Matrice di Proiezione "+ i +": ")
    print(p)
    i=i+1

# Salvo i risutati della calibrazione per un possibile utilizzo futuro (al momento non ci interessano rvecs / tvecs)
pickle.dump((cameraMatrix, projection_matrices[0]), open( "calibration.pkl", "wb" ))
pickle.dump(extrinsic_matrix, open( "extrMatrix.pkl", "wb" ))
pickle.dump(projection_matrices, open( "projMatrix.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))