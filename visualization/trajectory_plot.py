"""Plot interattivo HTML delle traiettorie GPS vs VO, in coordinate metriche locali.

Le due curve vivono nello stesso sistema (UTM con origine sul primo scatto),
quindi le differenze visibili sono drift puro della VO. Output: pagina HTML
con plotly, navigabile (zoom, pan, hover su ogni punto con indice frame).
"""

from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go


def _as_xy(points) -> np.ndarray:
    """Accetta lista di tuple, lista di UtmFrame o ndarray Nx2/Nx3."""
    if isinstance(points, np.ndarray):
        if points.ndim == 2 and points.shape[1] >= 2:
            return points[:, :2].astype(np.float64)
        raise ValueError(f"Shape inattesa per ndarray: {points.shape}")

    out = []
    for p in points:
        if hasattr(p, "east_rel") and hasattr(p, "north_rel"):
            out.append((p.east_rel, p.north_rel))
        elif hasattr(p, "__len__") and len(p) >= 2:
            out.append((float(p[0]), float(p[1])))
        else:
            raise TypeError(f"Punto non riconosciuto: {p!r}")
    return np.asarray(out, dtype=np.float64)


def _drift_finale(gps_xy: np.ndarray, vo_xy: np.ndarray) -> float | None:
    """Distanza fra l'ultimo punto GPS e l'ultimo punto VO (in metri)."""
    n = min(len(gps_xy), len(vo_xy))
    if n == 0:
        return None
    return float(np.linalg.norm(gps_xy[n - 1] - vo_xy[n - 1]))


def _hover_text(xy: np.ndarray, label: str) -> list[str]:
    return [
        f"<b>{label}</b><br>frame {i}<br>E={x:.2f} m<br>N={y:.2f} m"
        for i, (x, y) in enumerate(xy)
    ]


def plot_trajectories(gps_points,
                      vo_points,
                      output_path: str,
                      title: str = "Traiettoria GPS vs VO",
                      curve_indices=None) -> str:
    """Salva un HTML interattivo con GPS e VO sovrapposti.

    Parametri
    ----------
    gps_points : iterable di UtmFrame o (x, y) in metri
    vo_points  : iterable di (x, y) in metri (stessa origine UTM)
    output_path: percorso del file .html (cartelle create on-demand)
    curve_indices : indici opzionali dei frame "in curva", evidenziati con
        un cerchio rosso aperto sulla traccia VO.
    """
    gps_xy = _as_xy(gps_points)
    vo_xy = _as_xy(vo_points)

    fig = go.Figure()

    if len(gps_xy):
        fig.add_trace(go.Scatter(
            x=gps_xy[:, 0], y=gps_xy[:, 1],
            mode="lines+markers",
            name="GPS (UTM)",
            line=dict(color="royalblue", width=2),
            marker=dict(size=5, color="royalblue", symbol="circle"),
            text=_hover_text(gps_xy, "GPS"),
            hoverinfo="text",
        ))
        fig.add_trace(go.Scatter(
            x=[gps_xy[0, 0]], y=[gps_xy[0, 1]],
            mode="markers", name="GPS start",
            marker=dict(size=12, color="royalblue", symbol="square"),
            hoverinfo="skip",
        ))

    if len(vo_xy):
        fig.add_trace(go.Scatter(
            x=vo_xy[:, 0], y=vo_xy[:, 1],
            mode="lines+markers",
            name="VO stimata",
            line=dict(color="darkorange", width=2),
            marker=dict(size=6, color="darkorange", symbol="x"),
            text=_hover_text(vo_xy, "VO"),
            hoverinfo="text",
        ))
        fig.add_trace(go.Scatter(
            x=[vo_xy[0, 0]], y=[vo_xy[0, 1]],
            mode="markers", name="VO start",
            marker=dict(size=12, color="darkorange", symbol="square"),
            hoverinfo="skip",
        ))

    if curve_indices is not None and len(vo_xy):
        idx = np.asarray(list(curve_indices), dtype=int)
        idx = idx[(idx >= 0) & (idx < len(vo_xy))]
        if idx.size:
            fig.add_trace(go.Scatter(
                x=vo_xy[idx, 0], y=vo_xy[idx, 1],
                mode="markers",
                name="frame in curva (fallback GPS/IMU)",
                marker=dict(size=14, color="rgba(0,0,0,0)",
                            line=dict(color="red", width=1.5)),
                text=[f"<b>curva</b><br>frame {i}" for i in idx],
                hoverinfo="text",
            ))

    drift = _drift_finale(gps_xy, vo_xy)
    if drift is not None:
        title = f"{title}  -  drift finale: {drift:.2f} m"

    fig.update_layout(
        title=title,
        xaxis_title="Est rispetto al primo scatto [m]",
        yaxis_title="Nord rispetto al primo scatto [m]",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=10, r=10, b=10, t=60),
    )
    # Aspect ratio 1:1 cosi' la forma della traiettoria non e' deformata.
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.write_html(output_path, include_plotlyjs="cdn")
    return output_path


def plot_gps_only(gps_points, output_path: str,
                  title: str = "Traiettoria GPS") -> str:
    """Plot rapido della sola traccia GPS, utile per validare i metadati."""
    return plot_trajectories(gps_points, [], output_path=output_path, title=title)
