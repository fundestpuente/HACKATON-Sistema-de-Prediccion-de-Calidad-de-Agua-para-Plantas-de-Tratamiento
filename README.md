# Sistema de Predicción de Calidad de Agua para Plantas de Tratamiento
**Una herramienta de Machine Learning para evaluar la potabilidad del agua**

**Curso:** Samsung Innovation Campus – Inteligencia Artificial (Ecuador 2025)  
**Carpeta:** `/HACKATON-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento`

---

## 👥 Integrantes del Grupo
- Josue Malla
- Paul Altafuya
- Vladimir Espinoza 
- Patricio Quishpe

---

## 📝 Descripción del Proyecto
El acceso a agua potable segura es esencial para la salud pública y el desarrollo sostenible. Este proyecto desarrolla un **sistema inteligente de predicción de potabilidad del agua** utilizando algoritmos de Machine Learning, Visión por Computadora e Inteligencia Artificial Generativa.

El sistema analiza características físico-químicas críticas (pH, dureza, sólidos disueltos, cloraminas, sulfatos, conductividad, carbono orgánico, trihalometanos y turbidez) para determinar si una muestra de agua es segura.

### 🚀 Características Principales
- **Predicción ML:** Modelo entrenado para clasificar agua como Potable o No Potable.
- **Dashboard Interactivo:** Desarrollado en Streamlit para visualización y control.
- **Alertas en Tiempo Real:** Integración con **Telegram** para notificaciones automáticas de riesgo.
- **Asistente IA:** Chatbot integrado (OpenAI/Gemini) para consultas técnicas sobre calidad del agua.
- **Visión por Computadora:** Módulo experimental para análisis visual de turbidez.

---

## 📸 Imágenes Destacadas del Dashboard

*(Espacio reservado para capturas de pantalla del sistema)*

| Dashboard Principal | Predicción y Alertas |
|---------------------|----------------------|
| ![Dashboard](https://via.placeholder.com/400x200?text=Dashboard+Principal) | ![Prediccion](https://via.placeholder.com/400x200?text=Prediccion+y+Alertas) |

| Asistente IA | Análisis de Visión |
|--------------|--------------------|
| ![Chatbot](https://via.placeholder.com/400x200?text=Asistente+IA) | ![Vision](https://via.placeholder.com/400x200?text=Vision+Module) |

---

## ⚙️ Instrucciones de Instalación y Ejecución

### Requisitos Previos
- **Python 3.10+**
- **Cuenta de Telegram** (para las alertas)
- **API Key (Opcional):** OpenAI o Google Gemini para el chatbot.

### 🪜 Pasos de Instalación

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/fundestpuente/HACKATON-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento.git
   cd "HACKATON-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento"
   ```

2. **Crear y activar un entorno virtual:**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar Variables de Entorno:**
   Crea un archivo `.env` en la raíz del proyecto y añade tu token de Telegram:
   ```env
   TELEGRAM_TOKEN=tu_token_aqui
   ```
   
   **Notas:**
   - Para Telegram: Obtén tu token desde [@BotFather](https://t.me/botfather)
   - Para OpenAI: Obtén tu API key desde [platform.openai.com](https://platform.openai.com/api-keys)
   - La funcionalidad de Análisis de Imágenes requiere OpenAI API Key

5. **Ejecutar la aplicación:**
   ```bash
   streamlit run app.py
   ```
   El bot de Telegram se iniciará automáticamente en segundo plano.

---

## 🤖 Guía de Uso

### 1. Bot de Telegram
- Busca tu bot en Telegram y envía `/start`.
- En el Dashboard, usa el botón **"Sincronizar con Telegram"** en la barra lateral.
- Recibirás alertas si el agua es **NO POTABLE** o si el **pH** es inseguro.

### 2. Asistente IA
- Ve a la sección **"🤖 Asistente IA"**.
- Selecciona tu proveedor (OpenAI, Gemini, etc.).
- Pregunta sobre normativas, tratamientos o interpretación de datos.

### 3. Módulo de Visión (Experimental)
- Permite analizar imágenes de muestras de agua para estimar turbidez visualmente (requiere configuración de cámara o carga de imágenes).



## 📂 Estructura del Proyecto
```
SIC25-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento/
│
├── data/                       # Datos del proyecto
│   ├── processed/              # Datos limpios
│   ├── raw/                    # Datos crudos
│   └── test/                   # Muestras de prueba
│
├── models/                     # Modelos entrenados (.pkl)
│
├── notebooks/                  # Notebooks de Jupyter
│   ├── 01_eda_analisis.ipynb
│   ├── 02_limpieza_etl.ipynb
│   └── 03_entrenamiento.ipynb
│
├── src/                        # Código fuente
│   ├── chatbot_llm.py          # Lógica del Chatbot IA
│   ├── model_train.py          # Entrenamiento del modelo
│   ├── preprocessing.py        # Pipeline de preprocesamiento
│   ├── telegram_bot.py         # Bot de Telegram
│   ├── test_data.py            # Generador de datos dummy
│   └── vision_module.py        # Análisis de imágenes (Turbidez)
│
├── app.py                      # Aplicación principal (Streamlit)
├── requirements.txt            # Dependencias
├── telegram_connection.json    # Persistencia de usuarios del bot
└── README.md                   # Documentación
```

---

## Tecnologías Utilizadas
- **Core:** Python 3.10+
- **Interfaz:** Streamlit
- **ML/Data:** Scikit-learn, XGBoost, Pandas, Numpy
- **Visión:** OpenCV
- **IA Generativa:** OpenAI API, Google GenAI
- **Notificaciones:** Python-telegram-bot
- **Visualización:** Plotly, Matplotlib, Seaborn