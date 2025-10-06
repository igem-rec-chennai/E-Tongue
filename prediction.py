import pandas as pd
import numpy as np
import serial
from dash import Dash, html, dcc, Output, Input, State
from tensorflow.keras.models import load_model
import joblib

SERIAL_PORT = 'COM3'
BAUD_RATE = 9600
N_READINGS = 20
MODEL_PATH = 'saltiness_cnn_lstm_model.h5'
SCALER_PATH = 'saltiness_scaler.pkl'

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    print(f"✅ Connected to ESP32 on {SERIAL_PORT}")
except Exception as e:
    print(f"❌ Could not connect to ESP32: {e}")
    ser = None

def adc_to_mv(adc_val):
    return (adc_val / 4095) * 3300

def adc_to_current_uA(adc_val):
    voltage = adc_to_mv(adc_val)
    shunt_resistance = 1000
    return voltage / shunt_resistance * 1e3

def adc_to_conductivity_mS(adc_val):
    voltage = adc_to_mv(adc_val)
    return 0.0012 * voltage + 0.5

app = Dash(__name__)
app.title = "SaltEnPep e-Tongue"

app.layout = html.Div([
    html.H1("📊 SaltEnPep's e-tongue Live Monitoring",
            style={'textAlign':'center','color':'#8B4513','marginBottom':'30px','fontFamily':'Arial'}),
    html.Div([
        html.Button("▶️ Start Monitoring", id="start-button", n_clicks=0,
                    style={'fontSize': '20px','padding': '10px 20px',
                           'backgroundColor':'#28a745','color':'white','border':'none',
                           'borderRadius':'5px','cursor':'pointer','fontFamily':'Arial'})
    ], style={'textAlign':'center','marginBottom':'20px'}),
    dcc.Store(id='data-store', data=[]),
    dcc.Store(id='started', data=False),
    dcc.Interval(id='interval-component', interval=1500, n_intervals=0),
    html.H2("Live Readings Table", style={'textAlign':'center','marginTop':'20px','fontFamily':'Arial'}),
    html.Div(id='live-table', style={'margin':'auto'}),
    html.H2("Final Aggregate (Average)", style={'textAlign':'center','marginTop':'30px','fontFamily':'Arial'}),
    html.Div(id='aggregate-table', style={'margin':'auto'})
], style={'backgroundColor':'white','padding':'20px','fontFamily':'Arial'})

@app.callback(
    Output('started', 'data'),
    Input('start-button', 'n_clicks')
)
def start_monitor(n_clicks):
    return n_clicks > 0

@app.callback(
    Output('live-table', 'children'),
    Output('aggregate-table', 'children'),
    Output('data-store', 'data'),
    Input('interval-component', 'n_intervals'),
    State('data-store', 'data'),
    State('started', 'data')
)
def update_readings(n, data_store, started):
    if not started or ser is None:
        return html.Div(), html.Div(), data_store
    if data_store is None:
        data_store = []

    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            return html.Div(), html.Div(), data_store
        parts = line.split(',')
        if len(parts) != 4:
            print(f"⚠️ Invalid data: {line}")
            return html.Div(), html.Div(), data_store

        P1_ADC, P5_ADC, A2_ADC, C1_ADC = map(int, parts)

        P1_mV = adc_to_mv(P1_ADC)
        P5_mV = adc_to_mv(P5_ADC)
        DeltaE_mV = P1_mV - P5_mV
        A2_uA = adc_to_current_uA(A2_ADC)
        C1_mS = adc_to_conductivity_mS(C1_ADC)

        features = np.array([[P1_mV, P5_mV, DeltaE_mV, A2_uA, C1_mS]])
        features_scaled = scaler.transform(features)
        features_scaled = features_scaled.reshape((1, 1, features_scaled.shape[1]))

        pred_saltiness = model.predict(features_scaled, verbose=0)[0][0]

        new_row = {
            'Reading': len(data_store)+1,
            'P1_Potential_mV': round(P1_mV,3),
            'C1_Conductivity_mS': round(C1_mS,3),
            'A2_Current_uA': round(A2_uA,3),
            'Saltiness_Score': round(pred_saltiness,3)
        }
        data_store.append(new_row)

    except Exception as e:
        print(f"Error reading serial: {e}")
        return html.Div(), html.Div(), data_store

    df = pd.DataFrame(data_store)
    cols_to_show = ['Reading','P1_Potential_mV','C1_Conductivity_mS','A2_Current_uA','Saltiness_Score']
    col_widths = {
        'Reading': '80px',
        'P1_Potential_mV': '120px',
        'C1_Conductivity_mS': '130px',
        'A2_Current_uA': '110px',
        'Saltiness_Score': '120px'
    }

    live_table = html.Table([
        html.Thead(html.Tr([html.Th(col,
            style={'textAlign':'center','backgroundColor':'#8B3C48','color':'white','padding':'4px','width':col_widths[col]})
            for col in cols_to_show])),
        html.Tbody([html.Tr([html.Td(df.iloc[i][col],
                style={'textAlign':'center',
                       'backgroundColor':'#A67C52' if col=='Saltiness_Score' else '#F5F0E1',
                       'color':'white' if col=='Saltiness_Score' else 'black',
                       'padding':'3px','width':col_widths[col]})
                for col in cols_to_show]) for i in range(len(df))
        ])
    ], style={'width':'fit-content','border':'1px solid #D9CBB6','borderCollapse':'collapse','borderRadius':'5px','margin':'auto'})

    agg_table = html.Div()
    if len(data_store) >= N_READINGS:
        df_agg = df[cols_to_show[1:]].mean().to_frame().T
        agg_table = html.Table([
            html.Thead(html.Tr([html.Th(col,
                style={'textAlign':'center','backgroundColor':'#8B3C48','color':'white','padding':'4px','width':col_widths[col]})
                for col in cols_to_show[1:]])),
            html.Tbody([html.Tr([html.Td(round(df_agg.iloc[0][col],3),
                style={'textAlign':'center',
                       'backgroundColor':'#A67C52' if col=='Saltiness_Score' else '#EFE3D9',
                       'color':'white' if col=='Saltiness_Score' else 'black',
                       'padding':'3px','width':col_widths[col]})
                for col in cols_to_show[1:]])])
        ], style={'width':'fit-content','border':'1px solid #D9CBB6','borderCollapse':'collapse','borderRadius':'5px','margin':'auto'})

    return live_table, agg_table, data_store

if __name__ == '__main__':
    app.run(debug=False)
