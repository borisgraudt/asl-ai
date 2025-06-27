import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import threading
import socket
import json
import dash_daq as daq

# --- Архитектура сети ---
LAYER_NAMES = ['Input', 'Dense_1', 'Dense_2', 'Dense_3', 'Output']
LAYER_SIZES = [63, 256, 128, 64, 26]
LAYER_COLORS = ['#bfc7d5', '#4a90e2', '#4a90e2', '#4a90e2', '#ff9500']

# --- Глобальное хранилище активаций ---
latest_activations = [0.0 for _ in LAYER_NAMES]

# --- Dash App ---
app = dash.Dash(__name__)
app.title = 'Neural Network Visualization (Web)'

app.layout = html.Div([
    html.H2('Neural Network Visualization', style={'textAlign': 'center', 'fontFamily': 'San Francisco', 'marginTop': 20}),
    dcc.Graph(id='network-graph', config={'displayModeBar': False}, style={'height': '600px'}),
    html.Div([
        daq.GraduatedBar(
            id='legend-bar',
            color={"gradient":True,"ranges":{"#e6eaf2":[0,0.5],"#4a90e2":[0.5,1]}},
            min=0, max=1, value=0.5, showCurrentValue=False, step=0.01, size=300
        ),
        html.Span('Activation', style={'marginLeft': 16, 'fontFamily': 'San Francisco', 'fontSize': 18})
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginTop': 10})
], style={'backgroundColor': '#f8f8fa', 'height': '100vh'})

# --- Генерация графа сети ---
def make_network_figure(activations):
    n_layers = len(LAYER_NAMES)
    x = np.arange(n_layers) * 2
    y_max = max(LAYER_SIZES)
    traces = []
    # Нейроны (кружки)
    for i, (name, size, color) in enumerate(zip(LAYER_NAMES, LAYER_SIZES, LAYER_COLORS)):
        y = np.linspace(y_max, 0, size+2)[1:-1]  # равномерно по вертикали
        act = activations[i]
        if isinstance(act, (list, np.ndarray)) and len(np.shape(act)) > 0:
            if len(act) > 0:
                if isinstance(act, np.ndarray):
                    mean_act = float(np.mean(act))
                else:
                    mean_act = float(np.mean(np.array(act)))
            else:
                mean_act = 0.0
        else:
            mean_act = float(act)
        # Цвет нейронов — плавный градиент
        color_val = mean_act
        node_colors = [f'rgba(74,144,226,{0.25+0.75*color_val})' for _ in range(size)]
        traces.append(go.Scatter(
            x=[x[i]]*size,
            y=y,
            mode='markers',
            marker=dict(size=28, color=node_colors, line=dict(width=2, color=color)),
            text=[f'{name}<br>Activation: {mean_act:.3f}']*size,
            hoverinfo='text',
            showlegend=False
        ))
    # Стрелки между слоями
    for i in range(n_layers-1):
        traces.append(go.Scatter(
            x=[x[i]+0.5, x[i+1]-0.5],
            y=[y_max/2, y_max/2],
            mode='lines',
            line=dict(color='#b0b8c9', width=8, shape='spline'),
            hoverinfo='none',
            showlegend=False
        ))
    layout = go.Layout(
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, n_layers*2-1.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, y_max+2]),
        plot_bgcolor='#f8f8fa',
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode='closest',
        height=600,
        width=1100,
        shapes=[
            dict(
                type='rect',
                x0=x[i]-0.4, x1=x[i]+0.4,
                y0=0, y1=y_max,
                line=dict(color='rgba(0,0,0,0)'),
                fillcolor='rgba(0,0,0,0)'
            ) for i in range(n_layers)
        ]
    )
    return {'data': traces, 'layout': layout}

@app.callback(Output('network-graph', 'figure'), [Input('network-graph', 'id')])
def update_graph(_):
    return make_network_figure(latest_activations)

# --- Сервер для приёма активаций от main.py ---
def activation_server():
    global latest_activations
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 50007))
    while True:
        data, _ = sock.recvfrom(65536)
        try:
            acts = json.loads(data.decode())
            # acts: list of arrays for each layer (Dense_1, Dense_2, ...)
            # Для Input слоя берём 0.0
            acts_vis = [0.0]
            for act in acts:
                if isinstance(act, (list, np.ndarray)) and len(act) > 0:
                    mean = float(np.mean(act))
                else:
                    mean = 0.0
                acts_vis.append(mean)
            latest_activations = acts_vis
        except Exception as e:
            print('Ошибка при приёме активаций:', e)

threading.Thread(target=activation_server, daemon=True).start()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 