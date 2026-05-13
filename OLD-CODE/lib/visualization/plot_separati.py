import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
'''
def fit_dominant_quadrant(quadrante, yaw_angles):

    fitted_yaw_angles = []

    if quadrante =='Primo':
        return yaw_angles

    elif quadrante == 'Secondo':

        for yaw in yaw_angles:
            if yaw >0:
                yaw=180-yaw
            elif yaw<0:
                yaw = -(180+yaw)

            fitted_yaw_angles.append(yaw)

        return fitted_yaw_angles

    elif quadrante == 'Terzo':
        for yaw in yaw_angles:
            if yaw > 0:
                yaw = 180 - yaw
            elif yaw < 0:
                yaw = -(180 + yaw)

            fitted_yaw_angles.append(yaw)

        return fitted_yaw_angles


    elif quadrante == 'Quarto':

        for yaw in yaw_angles:
            if yaw > 0:
                yaw = 180 - yaw
            elif yaw < 0:
                yaw = -(180 + yaw)

            fitted_yaw_angles.append(yaw)

        return fitted_yaw_angles



def get_dominant_quadrant(pred_path):
    # Inizializza i contatori per ciascun quadrante
    quadrants_count = {"Primo": 0, "Secondo": 0, "Terzo": 0, "Quarto": 0}

    # Conta i punti in ciascun quadrante
    for x, y, _ in pred_path:
        if x > 0 and y > 0:
            quadrants_count["Primo"] += 1
        elif x < 0 and y > 0:
            quadrants_count["Secondo"] += 1
        elif x < 0 and y < 0:
            quadrants_count["Terzo"] += 1
        elif x > 0 and y < 0:
            quadrants_count["Quarto"] += 1

    # Determina il quadrante dominante
    dominant_quadrant = max(quadrants_count, key=quadrants_count.get)
    print("Il quadrante dominante è il: "+ str(dominant_quadrant))
    return dominant_quadrant

def calculate_yaw_from_coordinates(pred_path):

    quadrante = get_dominant_quadrant(pred_path)

    yaw_angles = []
    for i in range(len(pred_path)):
        
        if i == 0:
            x, y, _ = pred_path[i]
            yaw_start = np.arctan2(y - 0, x - 0)
            yaw_angles.append(np.degrees(yaw_start))  # Converte in gradi
        else:
            # Estrai le coordinate dei punti consecutivi
            #x1, y1, z1 = pred_path[i - 1]
            x2, y2, _ = pred_path[i]

            # Calcola l'angolo di yaw tra i due punti
            yaw = np.arctan2(y2 - y, x2 - x)  # Calcolo dell'angolo di yaw
            yaw = np.degrees(yaw)
            #yaw = normalize_yaw(yaw)
            yaw_angles.append(float(yaw))


    return fit_dominant_quadrant(quadrante, yaw_angles)
'''

def get_rotation_yaw(pred_yaw):
    yaw_iniziale = pred_yaw[0]

    # Calcola la differenza di yaw rispetto al punto iniziale, normalizza e converte in gradi
    yaw_rotations = [np.degrees((yaw - yaw_iniziale + np.pi) % (2 * np.pi) - np.pi) for yaw in pred_yaw]

    return yaw_rotations

def visualize_paths(gt_path, pred_path, pred_yaw, title="VO exercises", file_out= "plot_sep.html"): #, file_out="C:/Users/Asus/Desktop/Progetto_mosaicing/visualOdometry/calibrazione/data/plot.html"
    # Convertiamo le traiettorie in array numpy (se non lo sono già)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    yaw_rotations = get_rotation_yaw(pred_yaw)
    print("Rotazions yaw: ", yaw_rotations)
    #pred_yaw_calc= calculate_yaw_from_coordinates(pred_path)
    #print("pred_yaw (sep) content:", pred_yaw_calc)

    # Estrai le coordinate x, y, z
    gt_x, gt_y, gt_z = gt_path[:, 0], gt_path[:, 1], gt_path[:, 2]
    pred_x, pred_y, pred_z = pred_path[:, 0], pred_path[:, 1], pred_path[:, 2]

    # Calcola la differenza (errore) tra le due traiettorie
    diff = np.linalg.norm(gt_path - pred_path, axis=1)

    # Creazione della figura con 2 grafici 3D separati (subplots)
    fig = make_subplots(
        rows=1, cols=2,  # 1 riga e 2 colonne
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],  # specifica tipo 3D per entrambe le sottotrame
        subplot_titles=["Ground Truth Path", "Predicted Path"]  # Titoli per le sottotrame
    )

    # Traiettoria Ground Truth nel primo grafico
    fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode='lines+markers', name='GT',
                              marker=dict(size=4, color='blue', symbol='circle'), line=dict(color='blue', width=2)),
                  row=1, col=1)  # Aggiungi nella prima sottotrama

    hover_texts = [
        f"X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}<br>Rot: {yaw:.2f}°"
        for x, y, z, yaw in zip(pred_x, pred_y, pred_z, yaw_rotations)]

    # Traiettoria Predetta nel secondo grafico
    fig.add_trace(go.Scatter3d(x=pred_x, y=pred_y, z=pred_z, mode='lines+markers', name='Pred',
                              marker=dict(size=4, color='green', symbol='circle'), line=dict(color='green', width=2),hovertext=hover_texts, hoverinfo="text",),
                  row=1, col=2)  # Aggiungi nella seconda sottotrama

    # Configurazione del layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )

    # Salva il grafico in un file HTML
    fig.write_html(file_out)

    # Mostra il grafico
    fig.show()
