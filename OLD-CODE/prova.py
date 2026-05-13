import os
import numpy as np
import cv2
from tqdm import tqdm
import time
import pickle
import re
import glob
from lib.visualization import plot_one
from matplotlib import pyplot as plt
from utils.detect_feature_in_nxn_quadrants import detect_features_in_nxn_quadrants


def load_scales(filepath):
    scales = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip().startswith("Scale"):  # Controlla che la riga contenga 'Scale'
                # Estrai il valore numerico dopo 'Scale N:'
                parts = line.split(":")
                if len(parts) > 1:
                    scale_value = float(parts[1].strip())  # Converte in float
                    scales.append(scale_value)
    return scales

def load_calib(filepath):
    # caricamento dati matrice intrinseca della camera per rimuovere la distorsione

    try:
        with open(filepath, 'rb') as f:
            cameraMatrix, projMatrix = pickle.load(f)  # matrice della camera
            # print("cameraMatrix= ", cameraMatrix)
    except FileNotFoundError:
        print("Il file .pkl non esiste.")
    except pickle.UnpicklingError:
        print("Errore nel deserializzare il file Pickle.")

    # projMatrix utilizzato in triangulatePoints, nella funzione decompose_essential_mat

    return cameraMatrix, projMatrix

def carica_immagini(filepath, i):
    print(f"Numero di immagini da elaborare: {i}")
    immagini = glob.glob(filepath)
    immagini = sorted(immagini, key=lambda x: int(re.search(r'img_(\d+)', x).group(1)))[0:i]

    images = []
    for immagine in immagini:
        print("Leggendo immagine : " + str(immagine))
        img = cv2.imread(immagine)
        height, width = img.shape[:2]
        img = cv2.resize(img, (int(width / 2.5), int(height / 2.5)), interpolation=cv2.INTER_AREA)
        images.append(img)

    return images

def _form_transf(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.ravel()
    return T

def decomp_essential_mat2(E, q1, q2):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
    scores = []
    for R, t in candidates:
        T = _form_transf(R, t)
        P = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)
        dynamic_proj_matrix = K @ T[:3, :]
        hom_Q1 = cv2.triangulatePoints(dynamic_proj_matrix, P, q1.T, q2.T)
        hom_Q2 = T @ hom_Q1
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]
        score = sum(Q1[2, :] > 0) + sum(Q2[2, :] > 0)
        #relative_scale = np.mean(np.linalg.norm(Q1.T[:-1] - Q1.T[1:], axis=-1) /np.linalg.norm(Q2.T[:-1] - Q2.T[1:], axis=-1))
        scores.append(score)# + relative_scale)

    R, t = candidates[np.argmax(scores)]
    #T = self._form_transf(R, t)
    #P = np.matmul(np.concatenate((self.K, np.zeros((3, 1))), axis=1), T)
    #self.P = P
    return R, t


def decomp_essential_mat(E, K, q1, q2, prev_R=None):
    R1, R2, t = cv2.decomposeEssentialMat(E)
    candidates = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    best_R, best_t = None, None
    best_score = -1

    for R, t in candidates:
        T = _form_transf(R, t)
        P = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        dynamic_proj_matrix = K @ T[:3, :]

        hom_Q1 = cv2.triangulatePoints(dynamic_proj_matrix, P, q1.T, q2.T)
        hom_Q2 = T @ hom_Q1
        Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        score = np.sum(Q1[2, :] > 0) + np.sum(Q2[2, :] > 0)

        # 💡 Aggiungiamo un criterio di coerenza con la rotazione precedente
        if prev_R is not None:
            rotation_diff = np.linalg.norm(R - prev_R)
            score -= rotation_diff * 100  # Penalizza scelte con rotazione anomala

        if score > best_score:
            best_score = score
            best_R, best_t = R, t

    return best_R, best_t


data_dir = "C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/"
m=12
K, _ = load_calib(os.path.join(data_dir, 'data/calibration.pkl'))
scales= load_scales(os.path.join(data_dir, 'data/scales.txt'))
immagini = carica_immagini(os.path.join(data_dir, 'immagini/immagini_drone/immagini_senza_distorsione/*jpg'),m)
orb = cv2.ORB_create(5000)
cur_pose = np.eye(4)
estimated_path=[]
estimated_path.append((cur_pose[0, 3], cur_pose[1, 3]))

FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
prev_r = np.eye(3)

for i in range(1,len(immagini)):
    img1 = immagini[i-1]
    img2 = immagini[i]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    n=5

    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    kp1, des1 = detect_features_in_nxn_quadrants(img1, feature_path=os.path.join(data_dir, 'immagini/immagini_drone/feature/'), n=n,flag=False,i=i)
    kp2, des2 = detect_features_in_nxn_quadrants(img2, feature_path=os.path.join(data_dir, 'immagini/immagini_drone/feature/'), n=n,flag=False, i=i)

    if len(kp1) > 6 and len(kp2) > 6:
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.5 * n.distance]
        q1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        q2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

    _, mask = cv2.findHomography(q1, q2, cv2.RANSAC, 3.0)

    if mask is not None:
        mask = mask.ravel().astype(bool)
        q1 = q1[mask]
        q2 = q2[mask]

    E, _ = cv2.findEssentialMat(q1, q2, K)

    #R, t = decomp_essential_mat(E,K,q1,q2, prev_r)
    #R, t = decomp_essential_mat2(E, q1, q2)
    _, R, t, mask_pose = cv2.recoverPose(E, q1, q2, K)
    t = t * scales[i]
    transf = _form_transf(R,t)
    cur_pose = cur_pose @ transf
    estimated_path.append((cur_pose[0, 3], cur_pose[1, 3]))
    print(f"Frame {i}: Pose - X: {cur_pose[0, 3]}, Y: {cur_pose[1, 3]}")
file_out=f"C:/Users/Asus/Desktop/Progetto_mosaicing/visualOdometry/calibrazione/data/plot_est_2D_{m}.html"
file_out1=f"C:/Users/Asus/Desktop/plot_est_2D_{m}.html"
file_out = str(file_out1)
plot_one.visualize_path_2d(estimated_path, "percorso 2D stimato", file_out=file_out1)

