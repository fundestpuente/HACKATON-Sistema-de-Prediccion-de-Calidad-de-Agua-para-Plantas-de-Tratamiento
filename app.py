import streamlit as st
import threading
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import json 
import sys
import os
import datetime



# Añadir src al path para poder importar
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.telegram_bot import send_telegram_alert, run_listener
from src.vision_module import analyze_water_turbidity, get_ntu_interpretation
from src.chatbot_llm import create_chatbot_widget

#@st.cache_resource
def iniciar_bot_en_background():
    """
    Esta función crea un hilo secundario para correr el bot.
    Al usar @st.cache_resource, Streamlit asegura que esto solo se ejecute
    UNA vez al arrancar la app, evitando duplicar bots.
    """
    # Creamos el hilo apuntando a la función run_listener
    bot_thread = threading.Thread(target=run_listener, daemon=True)
    bot_thread.start()
    return bot_thread

# Llamamos a la función inmediatamente
# iniciar_bot_en_background()

# Configuración inicial
st.set_page_config(
    page_title="SIPCA",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300..700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');
    
    :root {
        --primary: var(--primary-color, #0c67a3);
        --accent: #11a4d4;
        --background: var(--background-color, #f0f4f8);
        --card: var(--secondary-background-color, #ffffff);
        --text-primary: var(--text-color, #101d22);
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

    /* Sliders: Thumb (círculo) blanco con borde azul */
    div[data-baseweb="slider"] div[role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid var(--primary) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        height: 18px !important; 
        width: 18px !important;
    }
    
    /* Sliders: Track Fill (Barra rellena) - Forzar color */
    div[data-baseweb="slider"] > div > div > div > div {
        background-color: var(--primary) !important;
    }
    
    /* Botón Secundario (Reset): Fondo gris claro */
    button[kind="secondary"] {
        background-color: #f1f5f9 !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        transition: all 0.2s;
    }
    button[kind="secondary"]:hover {
        background-color: #e2e8f0 !important;
        color: var(--text-primary) !important;
    }
    
    /* Botón Primario (Analizar): Azul con hover */
    button[kind="primary"] {
        background-color: var(--primary) !important;
        border: none !important;
        color: white !important;
        transition: all 0.2s;
    }
    button[kind="primary"]:hover {
        background-color: var(--accent) !important;
        box-shadow: 0 4px 12px rgba(12, 103, 163, 0.2);
    }

    /* Ocultar menú de Streamlit y footer */
    /*
    [data-testid="stToolbar"] {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    */

    /* --- NAVEGACIÓN SIDEBAR CON ICONOS --- */
    
    /* Ajuste del texto para alinear con el icono */
    section[data-testid="stSidebar"] div[role="radiogroup"] label p {
        font-size: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem; /* Espacio entre icono y texto */
    }

    /* INYECCIÓN DE ICONOS MATERIAL SYMBOLS */
    
    /* 1. Dashboard General */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1) p::before {
        content: "dashboard";
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        font-variation-settings: 'FILL' 0;
    }
    /* Relleno cuando está activo */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1):has(input:checked) p::before {
        font-variation-settings: 'FILL' 1;
        color: var(--primary);
    }
    
    /* 2. Visión por Computadora */
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2) p::before {
        content: "image_search";
        font-family: 'Material Symbols Outlined';
        font-size: 20px;
        font-weight: normal;
        font-variation-settings: 'FILL' 0;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2):has(input:checked) p::before {
        font-variation-settings: 'FILL' 1;
        color: var(--primary);
    }
    
</style>
""", unsafe_allow_html=True)



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
<div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; justify-content: center; width: 48px; height: 48px; background-color: #dbeafe; border-radius: 50%; color: #0369a1;">
        <span class="material-symbols-outlined" style="font-size: 28px; font-variation-settings: 'FILL' 1;">water_drop</span>
    </div>
    <div>
        <h1 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #0f172a; line-height: 1.2; font-family: 'Space Grotesk', sans-serif;">SIPCA</h1>
        <p style="margin: 0; font-size: 0.875rem; color: #64748b; font-weight: 500;">Predicción de potabilidad</p>
    </div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown('---')

# Navegación
options = ["Dashboard General", "Análisis de Imágenes"]
selection = st.sidebar.radio("Navegación", options, label_visibility="collapsed")
st.sidebar.markdown('---')

def tab_dashboard():
    # Definir los sliders con valores realistas o promedio
    def user_input_features():
        """Función para capturar los inputs del usuario a través de sliders"""
        st.sidebar.markdown('### Parámetros de la Muestra')
        
        # Agrupar parámetros para ahorrar espacio
        with st.sidebar.expander("Parámetros Básicos", expanded=True):
            ph = st.slider('pH', 0.0, 14.0, 7.0, 0.1)
            hardness = st.slider('Dureza (mg/L)', 50.0, 350.0, 196.0, 1.0)
            solids = st.slider('Sólidos (ppm)', 300.0, 60000.0, 22000.0, 100.0)
            chloramines = st.slider('Cloraminas (ppm)', 0.0, 14.0, 7.1, 0.1)

        with st.sidebar.expander("Parámetros Avanzados", expanded=False):
            sulfate = st.slider('Sulfato (mg/L)', 100.0, 500.0, 333.0, 1.0)
            conductivity = st.slider('Conductividad (µS/cm)', 100.0, 800.0, 420.0, 1.0)
            organic_carbon = st.slider('Carbono Orgánico (ppm)', 0.0, 30.0, 14.5, 0.1)
            trihalomethanes = st.slider('Trihalometanos', 0.0, 125.0, 66.0, 0.1)
            turbidity = st.slider('Turbidez', 1.0, 7.0, 3.9, 0.1)

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

    st.sidebar.markdown('---')
    with st.sidebar.expander("🔔 Conectar Alertas", expanded=True):
        # Enlace directo a tu bot
        bot_name = "AquaAlert_ec_Bot" # Pon el nombre real de tu bot sin @
        st.markdown(f"1. [Abrir Bot en Telegram](https://t.me/{bot_name}) y dar **/start**")
        
        if st.button("🔄 Sincronizar con Bot"):
            try:
                with open("telegram_connection.json", "r") as f:
                    data = json.load(f)
                
                # Guardar en sesión
                st.session_state['tg_id'] = data['chat_id']
                st.session_state['tg_name'] = data['name']
                st.success(f"Conectado: {data['name']}")
            except FileNotFoundError:
                st.warning("Primero ve a Telegram y usa /start")
                
        # Estado actual
        if 'tg_id' in st.session_state:
            st.caption(f"✅ Enviando a: {st.session_state['tg_name']}")
        else:
            st.caption("🔴 No conectado")
            
    # Botones de la barra lateral
    # st.sidebar.markdown('---')
    analyze_button = st.sidebar.button("Analizar Muestra", type="primary", use_container_width=True)
    st.sidebar.button("Restablecer Parámetros", type="secondary", use_container_width=True)

    # Área principal
    # st.title("Dashboard")

    # Bloque de análisis por lotes con icono Material Symbols
    with st.container(border=True):
        col_icon, col_text = st.columns([1, 15])
        with col_icon:
            st.markdown('<span class="material-symbols-outlined" style="font-size: 32px; color: var(--primary);">csv</span>', unsafe_allow_html=True)
        with col_text:
            st.markdown("### Análisis por lotes")
            st.caption("Sube un archivo CSV para realizar predicciones masivas. (Asegúrate de que las columnas coincidan con las esperadas a la muestra.)")
        
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
                
                st.subheader("Preview de Resultados")
                st.dataframe(batch_df)

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
        
        prediction = model.predict(input_scaled)[0]
        ph_val = input_df['ph'].iloc[0]
        
        # === NUEVO: GUARDAR ESTADO PARA EL BOT (/status) ===
        status_data = {
            "prediction": "POTABLE" if prediction == 1 else "NO POTABLE",
            "ph": float(ph_val),
            "confidence": float(confidence),
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
        }
        with open("water_status.json", "w") as f:
            json.dump(status_data, f)
        
        # SISTEMA DE ALERTAS INTEGRADO
        trigger = False
        reasons = []
        
        # 1. Criterio IA
        if prediction == 0: # 0 = No Potable
            trigger = True
            reasons.append(f"IA detectó riesgo (Confianza: {confidence:.1f}%)")
            
        # 2. Criterio Normativo (pH)
        ph_val = input_df['ph'].iloc[0]
        if ph_val < 6.5 or ph_val > 8.5:
            trigger = True
            reasons.append(f"pH fuera de norma ({ph_val:.1f})")

        # 3. Disparo de Alerta
        if trigger:
            # Recuperar ID de la sesión
            chat_id = st.session_state.get('tg_id')
            
            if chat_id:
                msg = (
                    f"🚨 *ALERTA DE CALIDAD DE AGUA*\n\n"
                    f"**Motivos:** {', '.join(reasons)}\n"
                    f"**Muestra:** pH {ph_val:.1f}"
                )
                # Llamamos a la función que importamos de src/telegram_bot.py
                ok, status = send_telegram_alert(msg, chat_id)
                
                if ok:
                    st.toast(f"Alerta enviada a {st.session_state['tg_name']}", icon="📲")
                else:
                    st.error(f"Fallo Telegram: {status}")
            else:
                st.warning("⚠️ Riesgo detectado, pero no has sincronizado el Bot.")
        
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
            st.subheader("Importancia de Características")
            
            importance_values = [0.25, 0.19, 0.10, 0.09, 0.085, 0.08, 0.075, 0.07, 0.07]  # Valores ficticios de importancia
            df_imp = pd.DataFrame({
                'Característica': FEATURES_IMPORTANCE_ORDER,
                'Importancia': importance_values
            }).sort_values(by='Importancia', ascending=True)
            
            # Gráfico de barras horizontales con color accent del diseño
            st.bar_chart(df_imp, x='Importancia', y='Característica', color='#11a4d4', height=400)

        # Gráfico Radar Chart
        with col_radar:
            st.subheader("Muestra vs Promedios Seguros")
            
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
                name='Tu Muestra',
                line_color='#11a4d4',
                fillcolor='rgba(17, 164, 212, 0.4)'
            ))
            
            # Promedios seguros 
            fig.add_trace(go.Scatterpolar(
                r=safe_values,
                theta=categories,
                fill='none',
                name='Promedios Seguros',
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

def tab_vision():
    st.header("Análisis de Imágenes")
    st.caption("Análisis de turbidez mediante visión por computadora. Sube una imagen de tu muestra de agua.")
    
    # Información educativa en un expander
    with st.expander("ℹ️ ¿Qué es la Turbidez (NTU)?", expanded=False):
        st.markdown("""
        La **turbidez** mide la claridad del agua. Se expresa en **NTU** (Unidades Nefelométricas de Turbidez).
        
        Partículas suspendidas como arcilla, sedimentos, algas y microorganismos causan turbidez.
        
        **Estándares:**
        - 🌟 **0-1 NTU**: Excelente (agua cristalina)
        - ✅ **1-5 NTU**: Cumple con OMS (potable)
        - ⚠️ **5-10 NTU**: Aceptable (visible al ojo)
        - 🔶 **10-25 NTU**: Deficiente (requiere tratamiento)
        - 🚫 **>25 NTU**: No potable
        """)
    
    # Layout de dos columnas
    col_upload, col_info = st.columns([2, 1])
    
    with col_upload:
        # Zona de carga con diseño mejorado
        with st.container(border=True):
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <span class="material-symbols-outlined" style="font-size: 48px; color: var(--primary);">add_photo_alternate</span>
                <h3 style="margin: 0.5rem 0;">Sube una imagen de agua</h3>
                <p style="color: var(--text-secondary); font-size: 0.9rem;">Formatos: JPG, PNG, JPEG</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Seleccionar imagen",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
    
    with col_info:
        # Guía rápida
        st.markdown("""
        <div style="background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%); 
                    padding: 1.5rem; border-radius: 0.75rem; height: 100%;">
            <h4 style="margin-top: 0;">📸 Consejos de Fotografía</h4>
            <ul style="font-size: 0.9rem; line-height: 1.8;">
                <li>Usa luz natural uniforme</li>
                <li>Fondo blanco o neutro</li>
                <li>Recipiente transparente</li>
                <li>Enfoque nítido</li>
                <li>Sin reflejos directos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Leer los bytes de la imagen
        image_bytes = uploaded_file.read()
        
        # Mostrar imagen subida
        col_img, col_result = st.columns([1, 1])
        
        with col_img:
            st.subheader("Muestra Analizada")
            st.image(image_bytes, use_container_width=True, caption=uploaded_file.name)
        
        with col_result:
            st.subheader("Resultados del Análisis")
            
            # Botón de análisis
            if st.button("🔬 Analizar Turbidez", type="primary", use_container_width=True):
                with st.spinner("Analizando imagen..."):
                    # Llamar a la función de visión
                    result = analyze_water_turbidity(image_bytes)
                    
                    if result.get('error'):
                        st.error(result['message'])
                    else:
                        # Guardar resultado en session_state
                        st.session_state['vision_result'] = result
                        st.rerun()
        
        # Mostrar resultados si existen
        if 'vision_result' in st.session_state:
            result = st.session_state['vision_result']
            
            # Determinar estilo según el estado
            if result['status'] == 'safe':
                icon_class = "potable"
                icon_symbol = "water_drop"
                border_color = "#22c55e"
            elif result['status'] == 'acceptable':
                icon_class = "no-potable"
                icon_symbol = "opacity"
                border_color = "#f59e0b"
            else:
                icon_class = "no-potable"
                icon_symbol = "warning"
                border_color = "#ef4444"
            
            st.markdown("---")
            
            # Tarjeta de resultado principal
            st.markdown(f"""
            <div style="background: rgba(255, 255, 255, 0.6);
                        border-radius: 1rem;
                        border: 3px solid {border_color};
                        padding: 2rem;
                        text-align: center;
                        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
                        margin: 2rem 0;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 2rem;">
                    <div class="result-icon {icon_class}">
                        <span class="material-symbols-outlined">{icon_symbol}</span>
                    </div>
                    <div style="text-align: left;">
                        <h1 style="margin: 0; font-size: 3rem; font-weight: bold; color: {border_color};">{result['ntu']} NTU</h1>
                        <h3 style="margin: 0.5rem 0; color: var(--text-primary);">{result['classification']}</h3>
                        <p style="margin: 0; color: var(--text-secondary);">Confianza: {result['confidence']}%</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Recomendación
            st.info(result['recommendation'])
            
            # Indicador de cumplimiento OMS
            if result['meets_who_standards']:
                st.success("✅ Cumple con los estándares de la OMS (< 5 NTU)")
            else:
                st.warning("⚠️ No cumple con los estándares recomendados de la OMS")
            
            # Métricas detalladas en columnas
            st.markdown("### 📊 Análisis Detallado")
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.metric(
                    label="Brillo Promedio",
                    value=f"{result['color_profile']['brightness']:.1f}",
                    help="Luminosidad media de la imagen (0-255)"
                )
            
            with metric_cols[1]:
                st.metric(
                    label="Saturación",
                    value=f"{result['color_profile']['saturation']:.1f}",
                    help="Intensidad de color promedio"
                )
            
            with metric_cols[2]:
                st.metric(
                    label="Variabilidad",
                    value=f"{result['color_profile']['std_dev']:.1f}",
                    help="Desviación estándar de la imagen"
                )
            
            with metric_cols[3]:
                st.metric(
                    label="Índice Amarillo",
                    value=f"{result['color_profile']['yellow_index']:.1f}",
                    help="Componente de color amarillo/marrón"
                )
            
            # Gráfico de referencia NTU
            st.markdown("### 📈 Escala de Referencia NTU")
            
            # Crear gráfico de barras horizontal con la escala
            ntu_ranges = [
                {'label': 'Excelente\n(0-1)', 'value': 1, 'color': '#22c55e'},
                {'label': 'Muy Buena\n(1-5)', 'value': 4, 'color': '#84cc16'},
                {'label': 'Buena\n(5-10)', 'value': 5, 'color': '#eab308'},
                {'label': 'Aceptable\n(10-25)', 'value': 15, 'color': '#f59e0b'},
                {'label': 'Deficiente\n(25-50)', 'value': 25, 'color': '#ef4444'},
                {'label': 'Muy Turbia\n(>50)', 'value': 50, 'color': '#991b1b'}
            ]
            
            fig = go.Figure()
            
            # Agregar barras de referencia
            for i, range_data in enumerate(ntu_ranges):
                fig.add_trace(go.Bar(
                    y=[range_data['label']],
                    x=[range_data['value']],
                    orientation='h',
                    marker=dict(color=range_data['color']),
                    name=range_data['label'],
                    showlegend=False
                ))
            
            # Agregar marcador de tu muestra
            y_position = 0
            for i, range_data in enumerate(ntu_ranges):
                cumulative = sum(r['value'] for r in ntu_ranges[:i+1])
                if result['ntu'] <= cumulative:
                    y_position = i
                    break
            
            fig.add_trace(go.Scatter(
                x=[result['ntu']],
                y=[ntu_ranges[y_position]['label']],
                mode='markers',
                marker=dict(
                    size=20,
                    color='white',
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                name='Tu Muestra',
                showlegend=True
            ))
            
            fig.update_layout(
                barmode='overlay',
                xaxis_title='NTU',
                height=400,
                showlegend=True,
                legend=dict(x=0.8, y=0.95),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Botón para nuevo análisis
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Analizar Nueva Imagen", use_container_width=True):
                    del st.session_state['vision_result']
                    st.rerun()
            
            with col_btn2:
                # Exportar resultados
                result_text = f"""
=== ANÁLISIS DE TURBIDEZ ===
Imagen: {uploaded_file.name}
Fecha: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTADO:
Turbidez: {result['ntu']} NTU
Clasificación: {result['classification']}
Confianza: {result['confidence']}%

RECOMENDACIÓN:
{result['recommendation']}

Cumple OMS: {'Sí' if result['meets_who_standards'] else 'No'}
                """
                st.download_button(
                    label="📄 Exportar Reporte",
                    data=result_text,
                    file_name=f"turbidity_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        # Estado inicial sin imagen
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: var(--text-secondary);">
            <span class="material-symbols-outlined" style="font-size: 80px; opacity: 0.3;">image</span>
            <h3 style="opacity: 0.6;">Esperando imagen...</h3>
            <p>Sube una imagen para comenzar el análisis de turbidez</p>
        </div>
        """, unsafe_allow_html=True)


if selection == "Dashboard General":
    tab_dashboard()
elif selection == "Análisis de Imágenes":
    tab_vision()

create_chatbot_widget()