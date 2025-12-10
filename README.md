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

---

## ⚙️ Instrucciones de Instalación y Ejecución

### Requisitos
- **Python 3.10+**
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

4. **Ejecutar la aplicación web:**
   ```bash
   streamlit run app.py
   ```
   La aplicación se abrirá automáticamente en tu navegador predeterminado (usualmente en `http://localhost:8501`).

---

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
│   ├── telegram_bot.py         # Bot de notificaciones (opcional)
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
