import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_path_3d(path, title, file_out): #, file_out="C:/Users/Asus/Desktop/Progetto_mosaicing/visualOdometry/calibrazione/data/plot.html"
    # Convertiamo le traiettorie in array numpy (se non lo sono già)
    path = np.array(path)


    # Estrai le coordinate x, y, z
    x, y, z = path[:, 0], path[:, 1], path[:, 2]


    # Creazione della figura con 2 grafici 3D
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Traiettoria Ground Truth
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', name='path',
                              marker=dict(size=4, color='blue', symbol='circle'), line=dict(color='blue', width=2)))

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
    #fig.write_html(f"C:/Users/Asus/Desktop/Progetto_mosaicing/Mosaicing/Stitching/data/{file_out}")
    fig.write_html(file_out)
    # Mostra il grafico
    fig.show()

def visualize_path_2d(path, title, file_out):
    import numpy as np
    import plotly.graph_objects as go

    print(f"File di output ricevuto: {file_out}")

    # Convertiamo le traiettorie in array numpy (se non lo sono già)
    path = np.array(path)

    # Estrai le coordinate x e y
    x, y = path[:, 0], path[:, 1]

    # Genera i numeri di ordine per ogni punto
    point_numbers = [f"{i}" for i in range(len(x))]

    # Creazione della figura
    fig = go.Figure()

    # Traiettoria con i numeri dei punti
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines+markers+text',
        name='path',
        marker=dict(size=6, color='blue', symbol='circle'),
        line=dict(color='blue', width=2),
        text=point_numbers,  # Aggiungi i numeri come etichette
        textposition='top center'  # Posiziona le etichette sopra i punti
    ))

    # Configurazione del layout
    fig.update_layout(
        title=title,
        xaxis_title='X',
        yaxis_title='Y',
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Salva il grafico in un file HTML
    fig.write_html(file_out)

    # Mostra il grafico
    fig.show()