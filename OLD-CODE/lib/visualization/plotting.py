import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_yaw_from_coordinates(pred_path):
    yaw_angles = []
    for i in range(len(pred_path)):
        if i==0:
            yaw = 0
            yaw_angles.append(np.degrees(yaw))  # Converte in gradi
        else:
            # Estrai le coordinate dei punti consecutivi
            x1, y1, z1 = pred_path[i - 1]
            x2, y2, z2 = pred_path[i]

            # Calcola l'angolo di yaw tra i due punti
            yaw = np.arctan2(y2 - y1, x2 - x1)  # Calcolo dell'angolo di yaw
            yaw_angles.append(float(np.degrees(yaw)))  # Converte in gradi
    return yaw_angles

def visualize_paths(gt_path, pred_path, pred_yaw, title="VO exercises", file_out= "plotting.html"): #, file_out="C:/Users/Asus/Desktop/Progetto_mosaicing/visualOdometry/calibrazione/data/plot.html"
    # Convertiamo le traiettorie in array numpy (se non lo sono già)
    gt_path = np.array(gt_path)
    pred_path = np.array(pred_path)

    pred_yaw_calc = calculate_yaw_from_coordinates(pred_path)
    print("pred_yaw content:", pred_yaw_calc)

    # Estrai le coordinate x, y, z
    gt_x, gt_y, gt_z = gt_path[:, 0], gt_path[:, 1], gt_path[:, 2]
    pred_x, pred_y, pred_z = pred_path[:, 0], pred_path[:, 1], pred_path[:, 2]

    # Calcola la differenza (errore) tra le due traiettorie
    diff = np.linalg.norm(gt_path - pred_path, axis=1)

    # Creazione della figura con 2 grafici 3D
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Traiettoria Ground Truth
    fig.add_trace(go.Scatter3d(x=gt_x, y=gt_y, z=gt_z, mode='lines+markers', name='GT',
                              marker=dict(size=4, color='blue', symbol='circle'), line=dict(color='blue', width=2)))

    hover_texts = [
        f"X: {x:.2f}<br>Y: {y:.2f}<br>Z: {z:.2f}<br>Yaw: {yaw:.2f}°"
        for x, y, z, yaw in zip(pred_x, pred_y, pred_z, pred_yaw_calc)]

    # Traiettoria Predetta
    fig.add_trace(go.Scatter3d(x=pred_x, y=pred_y, z=pred_z, mode='lines+markers', name='Pred',
                              marker=dict(size=4, color='green', symbol='circle'), line=dict(color='green', width=2), hovertext=hover_texts, hoverinfo="text"))

    # Aggiungere le linee che mostrano l'errore (differenza tra GT e Pred)
    for i in range(len(gt_x)):
        fig.add_trace(go.Scatter3d(
            x=[gt_x[i], pred_x[i]], y=[gt_y[i], pred_y[i]], z=[gt_z[i], pred_z[i]],
            mode='lines', line=dict(color='red', dash='dash'), name='Error', showlegend=False
        ))

    # Configurazione del layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Salva il grafico in un file HTML
    fig.write_html(file_out)

    # Mostra il grafico
    fig.show()




