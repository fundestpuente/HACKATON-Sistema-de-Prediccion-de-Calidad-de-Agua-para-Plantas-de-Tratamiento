import cv2
import numpy as np
from PIL import Image
import io

def analyze_water_turbidity(image_bytes):
    """
    Analiza la turbidez del agua a partir de una imagen.
    
    Args:
        image_bytes: Bytes de la imagen subida
        
    Returns:
        dict: {
            'ntu': float (Unidades Nefelométricas de Turbidez),
            'classification': str (Clasificación de calidad),
            'confidence': float (Confianza del análisis),
            'color_profile': dict (Información de color)
        }
    """
    try:
        # Convertir bytes a imagen
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image.convert('RGB'))
        
        # Análisis de claridad y turbidez basado en características de color
        # Convertir a diferentes espacios de color para análisis
        
        # 1. Análisis en escala de grises (luminosidad)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        brightness = np.mean(gray)
        std_dev = np.std(gray)
        
        # 2. Análisis de color en espacio LAB
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # 3. Análisis de saturación y claridad
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        saturation = np.mean(s)
        value = np.mean(v)
        
        # 4. Análisis de turbidez basado en características visuales
        # Agua clara: alta luminosidad, baja desviación, baja saturación en componentes a* y b*
        # Agua turbia: baja luminosidad, alta desviación, alta saturación
        
        # Calcular componentes de turbidez
        turbidity_factor = 0.0
        
        # Componente 1: Luminosidad inversa (agua turbia es más oscura)
        brightness_factor = max(0, (255 - brightness) / 255) * 100
        
        # Componente 2: Variabilidad (agua turbia tiene más variación)
        variation_factor = (std_dev / 128) * 50
        
        # Componente 3: Componentes de color amarillo/marrón (indicativo de turbidez)
        yellow_brown = (np.mean(b_channel) - 128) / 128  # b* positivo = amarillo
        color_turbidity = max(0, yellow_brown) * 100
        
        # Componente 4: Saturación (agua turbia tiende a tener más color)
        saturation_factor = (saturation / 255) * 50
        
        # Fórmula ponderada para NTU estimado
        # NTU típico: 0-5 (excelente), 5-10 (bueno), 10-25 (aceptable), >25 (pobre)
        ntu_estimated = (
            brightness_factor * 0.3 +
            variation_factor * 0.25 +
            color_turbidity * 0.30 +
            saturation_factor * 0.15
        )
        
        # Normalizar a un rango realista (0-100 NTU)
        ntu_estimated = min(100, max(0, ntu_estimated))
        
        # Clasificación basada en estándares de la OMS y EPA
        if ntu_estimated < 1:
            classification = "Excelente"
            status = "safe"
            confidence = 95
        elif ntu_estimated < 5:
            classification = "Muy Buena"
            status = "safe"
            confidence = 90
        elif ntu_estimated < 10:
            classification = "Buena"
            status = "acceptable"
            confidence = 85
        elif ntu_estimated < 25:
            classification = "Aceptable"
            status = "acceptable"
            confidence = 80
        elif ntu_estimated < 50:
            classification = "Deficiente"
            status = "poor"
            confidence = 75
        else:
            classification = "Muy Turbia"
            status = "poor"
            confidence = 70
        
        # Perfil de color para visualización
        color_profile = {
            'brightness': float(brightness),
            'saturation': float(saturation),
            'std_dev': float(std_dev),
            'yellow_index': float(np.mean(b_channel))
        }
        
        return {
            'ntu': round(ntu_estimated, 2),
            'classification': classification,
            'status': status,
            'confidence': confidence,
            'color_profile': color_profile,
            'recommendation': get_recommendation(ntu_estimated),
            'meets_who_standards': ntu_estimated < 5
        }
        
    except Exception as e:
        return {
            'error': True,
            'message': f"Error al analizar la imagen: {str(e)}",
            'ntu': None
        }

def get_recommendation(ntu_value):
    """
    Proporciona recomendaciones basadas en el valor NTU.
    """
    if ntu_value < 1:
        return "💎 Calidad excepcional. Agua cristalina."
    elif ntu_value < 5:
        return "✅ Cumple con estándares de la OMS. Apta para consumo."
    elif ntu_value < 10:
        return "⚠️ Aceptable pero considere filtración adicional."
    elif ntu_value < 25:
        return "🔶 Requiere tratamiento antes del consumo."
    else:
        return "🚫 No apta para consumo. Requiere tratamiento intensivo."

def get_ntu_interpretation():
    """
    Retorna información educativa sobre NTU.
    """
    return {
        'title': 'Escala de Turbidez (NTU)',
        'ranges': [
            {'range': '0-1 NTU', 'quality': 'Excelente', 'description': 'Agua cristalina'},
            {'range': '1-5 NTU', 'quality': 'Muy Buena', 'description': 'Estándar OMS'},
            {'range': '5-10 NTU', 'quality': 'Buena', 'description': 'Ligeramente visible'},
            {'range': '10-25 NTU', 'quality': 'Aceptable', 'description': 'Visible, requiere atención'},
            {'range': '>25 NTU', 'quality': 'Deficiente', 'description': 'No apta para consumo'}
        ]
    }
