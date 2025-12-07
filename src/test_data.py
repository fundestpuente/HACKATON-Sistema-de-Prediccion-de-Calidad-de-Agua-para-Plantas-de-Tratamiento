import pandas as pd
import os

# Definir las columnas exactas que espera el dashboard
columns = [
    'ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
    'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity'
]

# Crear datos sintéticos
data = [
    # Caso 1: Valores optimizados (Probablemente Potable)
    [7.2, 190.0, 20000.0, 7.5, 330.0, 420.0, 14.0, 65.0, 3.8],
    
    # Caso 2: Valores extremos/contaminados (Probablemente No Potable)
    [11.5, 310.0, 55000.0, 2.1, 480.0, 750.0, 28.0, 110.0, 6.5],
    
    # Caso 3: Valores promedio del dataset
    [7.0, 196.0, 22014.0, 7.1, 333.0, 426.0, 14.3, 66.4, 3.9]
]

# Crear DataFrame
df_test = pd.DataFrame(data, columns=columns)

# Configurar rutas absolutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, '../data/test')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'test_samples.csv')

# Crear directorio si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Guardar como CSV
df_test.to_csv(OUTPUT_FILE, index=False)

print(f"Archivo creado exitosamente en: {OUTPUT_FILE}")
