# 💧 Sistema de Predicción de Calidad de Agua para Plantas de Tratamiento
**Una herramienta de Machine Learning para evaluar la potabilidad del agua**

**Curso:** Samsung Innovation Campus – Inteligencia Artificial (Ecuador 2025)  
**Carpeta:** `/SIC25-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento`

---

## 👥 Integrantes del Grupo
- Josue Malla
- Paul Altafuya
- Vladimir Espinoza 
- Patricio Quishpe

---

## 📝 Descripción del Proyecto
El acceso a agua potable segura es esencial para la salud pública y el desarrollo sostenible. La calidad del agua puede verse comprometida por diversos factores químicos y físicos que no siempre son detectables a simple vista.

Este proyecto tiene como objetivo desarrollar un **sistema inteligente de predicción de potabilidad del agua** utilizando algoritmos de Machine Learning. El modelo analiza características físico-químicas críticas como el pH, la dureza, los sólidos disueltos, las cloraminas, los sulfatos, la conductividad, el carbono orgánico, los trihalometanos y la turbidez para determinar si una muestra de agua es segura para el consumo humano.

La solución incluye un **dashboard interactivo desarrollado en Streamlit** que permite:
- Ingresar parámetros manualmente para una evaluación rápida.
- Cargar archivos CSV para realizar predicciones masivas (por lotes).
- Visualizar la importancia de las características y comparar la muestra con promedios seguros.
- **Recibir alertas en tiempo real vía Telegram** cuando se detectan niveles de riesgo o agua no potable.

---

## ⚙️ Instrucciones de Instalación y Ejecución

### Requisitos
- **Python 3.10+**
- **Cuenta de Telegram** (para las alertas)
- **Librerías:** incluidas en `requirements.txt`

### 🪜 Pasos de Ejecución

1. **Clonar el repositorio o ubicarte en la carpeta del proyecto:**
   ```bash
   git clone https://github.com/fundestpuente/SIC25-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento.git
   cd "SIC25-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento"
   ```

2. **Crear y activar un entorno virtual (opcional pero recomendado):**
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar Variables de Entorno:**
   Crea un archivo `.env` en la raíz del proyecto y añade tus credenciales:
   ```env
   TELEGRAM_TOKEN=tu_token_de_telegram_aqui
   OPENAI_API_KEY=tu_api_key_de_openai_aqui
   ```
   
   **Notas:**
   - Para Telegram: Obtén tu token desde [@BotFather](https://t.me/botfather)
   - Para OpenAI: Obtén tu API key desde [platform.openai.com](https://platform.openai.com/api-keys)
   - La funcionalidad de Análisis de Imágenes requiere OpenAI API Key

5. **Ejecutar la aplicación web:**
   Al iniciar la aplicación, el **Bot de Telegram se iniciará automáticamente** en segundo plano.
   ```bash
   streamlit run app.py
   ```
   La aplicación se abrirá en tu navegador y verás en la terminal el mensaje: `🤖 Bot de Alertas ESCUCHANDO...`

---

## 🤖 Uso del Bot de Telegram

1. Abre tu bot en Telegram.
2. Envía el comando `/start`. El bot registrará tu ID de usuario.
3. En el Dashboard de Streamlit, ve a la barra lateral y presiona el botón **"Sincronizar con Telegram"**.
4. ¡Listo! Recibirás alertas automáticas si:
   - La predicción indica que el agua es **NO POTABLE**.
   - El **pH** se encuentra fuera del rango seguro (6.5 - 8.5).

---
---

## 🤖 Asistente IA de Calidad de Agua

El sistema incluye un chatbot inteligente que puede responder preguntas sobre:
- Parámetros de calidad de agua y sus valores seguros
- Interpretación de resultados de análisis
- Normativas y estándares (OMS, EPA)
- Recomendaciones técnicas y procedimientos

### Proveedores Soportados:
- **OpenAI (GPT-3.5/GPT-4)** - Respuestas precisas y rápidas
- **Google Gemini** - ⭐ GRATIS, sin tarjeta de crédito
- **Anthropic (Claude)** - Excelente para explicaciones técnicas

### Configuración Rápida:
1. Obtén una API key de tu proveedor preferido 
2. En la app, ve a la barra lateral → "🤖 Asistente IA"
3. Selecciona tu proveedor e ingresa tu API key
4. ¡Empieza a hacer preguntas!



## 📂 Estructura del Código
```
SIC25-Sistema-de-Prediccion-de-Calidad-de-Agua-para-Plantas-de-Tratamiento/
│
├── data/                       # Conjuntos de datos
│   ├── processed/              # Datos limpios y procesados
│   ├── raw/                    # Datos originales (water_potability.csv)
│   └── test/                   # Muestras de prueba
│
├── models/                     # Modelos serializados y escaladores
│   ├── water_potability_model.pkl
│   └── scaler.pkl
│
├── notebooks/                  # Notebooks de Jupyter para análisis y experimentación
│   ├── 01_eda_analisis.ipynb   # Análisis Exploratorio de Datos (EDA)
│   ├── 02_limpieza_etl.ipynb   # Limpieza y transformación de datos
│   └── 03_entrenamiento.ipynb  # Entrenamiento y evaluación de modelos
│
├── src/                        # Código fuente modular
│   ├── model_train.py          # Script de entrenamiento
│   ├── preprocessing.py        # Funciones de preprocesamiento
│   ├── telegram_bot.py         # Bot de notificaciones y alertas
│   └── test_data.py            # Generación de datos de prueba
│
├── app.py                      # Aplicación principal (Dashboard Streamlit)
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Documentación del proyecto
```

---

## ✅ Herramientas Implementadas
- **Lenguaje:** Python 3.10+
- **Framework Web:** Streamlit
- **Machine Learning:** Scikit-learn, XGBoost, Imbalanced-learn
- **Análisis y Procesamiento:** Pandas, Numpy
- **Visualización:** Plotly, Matplotlib, Seaborn
- **Notificaciones:** Python-telegram-bot API
- **Control de Versiones:** Git + GitHub

---

## 🌱 Impacto del Proyecto

Este sistema contribuye a:

- **Automatizar la evaluación** de la calidad del agua en plantas de tratamiento.
- **Reducir el tiempo** de análisis mediante predicciones instantáneas.
- **Apoyar la toma de decisiones** con visualizaciones claras sobre los factores de riesgo.
- **Mejorar la salud pública** al identificar agua no potable antes de su distribución.

> "El agua es la fuerza motriz de toda la naturaleza."  
> — *Leonardo da Vinci*
