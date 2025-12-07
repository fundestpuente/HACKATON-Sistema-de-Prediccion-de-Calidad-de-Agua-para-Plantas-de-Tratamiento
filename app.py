import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# Configuración inicial
st.set_page_config(
    page_title="Water Potability Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
    
    /* Variables de color del diseño */
    :root {
        --primary: #0c67a3;
        --accent: #11a4d4;
        --background: #f0f4f8;
        --card: #ffffff;
        --text-primary: #101d22;
        --text-secondary: #5a6e79;
        --border-color: #e2e8f0;
    }
    
    /* Estilos generales */
    .stApp {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    /* Iconos Material Symbols */
    .material-symbols-outlined {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    
    /* Tarjeta de resultado personalizada */
    .result-card {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 1rem;
        border: 1px solid var(--border-color);
        padding: 3rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        margin: 2rem 0;
    }
    
    .result-icon {
        width: 96px;
        height: 96px;
        margin: 0 auto 1rem;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 60px;
    }
    
    .result-icon.potable {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
    }
    
    .result-icon.no-potable {
        background: rgba(239, 68, 68, 0.1);
        color: #ef4444;
    }
    
    .result-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .result-title.potable {
        color: #22c55e;
    }
    
    .result-title.no-potable {
        color: #ef4444;
    }
    
    .result-confidence {
        color: var(--text-secondary);
        font-size: 1.125rem;
    }
</style>
""", unsafe_allow_html=True)

import os

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/water_potability_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
# Cargar modelos y escalador

@st.cache_resource
def load_artifacts():
    """Carga el modelo y el escalador"""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception:
        st.error("Error: No se encontró el modelo o el escalador. Por favor, asegúrese de que los archivos existen en la ruta especificada.")
        return None, None

model, scaler = load_artifacts()

# Lista ordenada de variables por importancia
FEATURES_IMPORTANCE_ORDER = [
    'Sulfate', 'ph', 'Solids', 'Hardness', 'Chloramines',
    'Trihalomethanes', 'Turbidity', 'Conductivity', 'Organic_carbon',
]

# Sidebar con iconos Material Symbols
st.sidebar.markdown("""
<div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; justify-content: center; width: 40px; height: 40px; background-color: rgba(12, 103, 163, 0.1); border-radius: 0.5rem; color: #0c67a3;">
        <span class="material-symbols-outlined" style="font-size: 24px;">water_drop</span>
    </div>
    <div>
        <h1 style="margin: 0; font-size: 1.125rem; font-weight: bold; color: var(--text-primary);">Water Potability</h1>
        <p style="margin: 0; font-size: 0.875rem; color: var(--text-secondary);">Prediction Dashboard</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown('---')

# Definir los sliders con valores realistas o promedio
def user_input_features():
    """Función para capturar los inputs del usuario a través de sliders"""
    st.sidebar.markdown('### Input Parameters')
    
    # Agrupar parámetros para ahorrar espacio
    with st.sidebar.expander("Basic Parameters", expanded=True):
        ph = st.slider('pH', 0.0, 14.0, 7.0, 0.1)
        hardness = st.slider('Hardness (mg/L)', 50.0, 350.0, 196.0, 1.0)
        solids = st.slider('Solids (ppm)', 300.0, 60000.0, 22000.0, 100.0)
        chloramines = st.slider('Chloramines (ppm)', 0.0, 14.0, 7.1, 0.1)

    with st.sidebar.expander("Advanced Parameters", expanded=False):
        sulfate = st.slider('Sulfate (mg/L)', 100.0, 500.0, 333.0, 1.0)
        conductivity = st.slider('Conductivity (µS/cm)', 100.0, 800.0, 420.0, 1.0)
        organic_carbon = st.slider('Organic Carbon (ppm)', 0.0, 30.0, 14.5, 0.1)
        trihalomethanes = st.slider('Trihalomethanes', 0.0, 125.0, 66.0, 0.1)
        turbidity = st.slider('Turbidity', 1.0, 7.0, 3.9, 0.1)

    data = {
        'ph': ph,
        'Hardness': hardness,
        'Solids': solids,
        'Chloramines': chloramines,
        'Sulfate': sulfate,
        'Conductivity': conductivity,
        'Organic_carbon': organic_carbon,
        'Trihalomethanes': trihalomethanes,
        'Turbidity': turbidity
    }
    
    return pd.DataFrame([data], index=['Your Sample'])

input_df = user_input_features()

# Botones de la barra lateral
# st.sidebar.markdown('---')
analyze_button = st.sidebar.button("Analizar Muestra o CSV  ", type="primary")
st.sidebar.button("Restablecer Parámetros", type="secondary")

# Área principal
st.title("Dashboard")

# Bloque de análisis por lotes con icono Material Symbols
with st.container(border=True):
    col_icon, col_text = st.columns([1, 15])
    with col_icon:
        st.markdown('<span class="material-symbols-outlined" style="font-size: 32px; color: #0c67a3;">csv</span>', unsafe_allow_html=True)
    with col_text:
        st.markdown("### Análisis por lotes")
        st.caption("Sube un archivo CSV para realizar predicciones masivas. (Asegúrate de que las columnas coincidan con las esperadas.)")
    
    csv_file = st.file_uploader(" ", type=["csv"], label_visibility="collapsed")

if csv_file is not None:
    batch_df = pd.read_csv(csv_file)
    st.subheader("Preview de Archivo CSV")
    st.dataframe(batch_df.head())
    
    # Predicción de lotes
    if st.button("Ejecutar Predicción por Lotes", type="primary"):
        try:
            batch_scaled = scaler.transform(batch_df)
            predictions = model.predict(batch_scaled)
            batch_df['Potability_Prediction'] = np.where(predictions == 1, 'POTABLE', 'NO POTABLE')
            
            st.success("Análisis por lotes completado.")
            st.download_button(
                label="Descargar Resultados como CSV",
                data=batch_df.to_csv(index=False).encode('utf-8'),
                file_name='water_potability_results.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error al procesar el lote: {e}. Asegurate de que las columnas coinciden con las esperadas.")

# Predicción y resultados
if analyze_button and model:
    # Preprocesamiento
    input_scaled = scaler.transform(input_df)
    
    # Predicción
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    confidence = proba[prediction] * 100
    
    # Mostrar resultados con diseño del mockup
    if prediction == 1:
        icon_class = "potable"
        icon_symbol = "check_circle"
        title_text = "Potable"
        title_class = "potable"
    else:
        icon_class = "no-potable"
        icon_symbol = "cancel"
        title_text = "NO Potable"
        title_class = "no-potable"
    
    st.markdown(f"""
    <div class="result-card">
        <div class="result-icon {icon_class}">
            <span class="material-symbols-outlined">{icon_symbol}</span>
        </div>
        <h3 class="result-title {title_class}">{title_text}</h3>
        <p class="result-confidence">{confidence:.1f}% Confidence</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizaciones
    col_feat_imp, col_radar = st.columns([3, 2])
    
    # Gráfico1: Importancia de características
    with col_feat_imp:
        st.subheader("Feature Importance")
        
        importance_values = [0.25, 0.19, 0.10, 0.09, 0.085, 0.08, 0.075, 0.07, 0.07]  # Valores ficticios de importancia
        df_imp = pd.DataFrame({
            'Feature': FEATURES_IMPORTANCE_ORDER,
            'Importance': importance_values
        }).sort_values(by='Importance', ascending=True)
        
        # Gráfico de barras horizontales con color accent del diseño
        st.bar_chart(df_imp, x='Importance', y='Feature', color='#11a4d4', height=400)

    # Gráfico Radar Chart
    with col_radar:
        st.subheader("Sample vs Safe Averages")
        
        # Valores promedios seguros estimados para la comparación
        safe_avg_values = {
                'ph': 7.5, 'Hardness': 180.0, 'Solids': 15000.0, 'Chloramines': 5.0, 
                'Sulfate': 300.0, 'Conductivity': 500.0, 'Organic_carbon': 10.0, 
                'Trihalomethanes': 70.0, 'Turbidity': 4.0
        }
        
        # Asegurar el orden del sample coincida con el safe_avg
        sample_values = input_df[list(safe_avg_values.keys())].iloc[0].values.tolist()
        safe_values = list(safe_avg_values.values())
        categories = list(safe_avg_values.keys())
        
        # Encontrar el valor máximo para establecer el rango del eje polar
        max_val = max(max(sample_values), max(safe_values))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=sample_values,
            theta=categories,
            fill='toself',
            name='Your Sample',
            line_color='#11a4d4',
            fillcolor='rgba(17, 164, 212, 0.4)'
        ))
        
        # Promedios seguros 
        fig.add_trace(go.Scatterpolar(
            r=safe_values,
            theta=categories,
            fill='none',
            name='Safe Average',
            line=dict(dash='dot', color='#4ade80')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max_val * 1.1]),
                angularaxis=dict(tickfont=dict(size=10), direction = "clockwise")
            ),
            showlegend=True,
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, width="stretch")