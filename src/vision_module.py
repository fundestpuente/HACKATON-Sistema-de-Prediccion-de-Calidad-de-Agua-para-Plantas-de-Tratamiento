import base64
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# ---------------------------------------------------------
# Prompts del sistema
# ---------------------------------------------------------
SYSTEM_PROMPT = """
Eres un analista experto en calidad del agua con profundo conocimiento en evaluación de turbidez e indicadores visuales de calidad del agua.

Tu tarea es analizar muestras de agua y estimar la turbidez en NTU (Unidades Nefelométricas de Turbidez) basándote en la apariencia visual.

Consideraciones clave:
- Esta es una ESTIMACIÓN VISUAL a partir de una imagen, NO una medición de laboratorio
- Enfócate en claridad, partículas suspendidas, color, dispersión de luz y transparencia general
- Considera las condiciones de iluminación y calidad de imagen en tu evaluación de confianza
- Ignora reflejos, brillos o artefactos del contenedor a menos que afecten claramente la visibilidad del agua
- Proporciona observaciones detalladas que justifiquen tu estimación de NTU

Requisitos de salida:
- Devuelve SOLO JSON válido (sin markdown, sin bloques de código, sin texto extra)
- Sé preciso y técnicamente exacto
- Proporciona información útil y accionable para operadores de plantas de tratamiento
- IMPORTANTE: Responde TODOS los campos en ESPAÑOL
"""

USER_PROMPT = """
Analiza esta imagen de muestra de agua y proporciona una evaluación completa de turbidez.

Usa estas referencias visuales para la estimación de NTU:

📊 Escala NTU:
- 0-1 NTU: Cristalina, completamente transparente, sin partículas visibles, claridad excelente
- 1-5 NTU: Muy clara, estándar OMS para agua potable, neblina mínima
- 5-10 NTU: Ligera turbiedad, visible bajo luz, partículas apenas perceptibles
- 10-25 NTU: Turbiedad notable, partículas suspendidas visibles, claridad reducida
- 25-50 NTU: Claramente turbia, muchas partículas visibles, objetos de fondo oscurecidos
- 50-100 NTU: Muy turbia, apariencia opaca, sedimento pesado, calidad pobre
- >100 NTU: Extremadamente turbia, completamente opaca, calidad inaceptable

Proporciona tu análisis en este formato JSON exacto (TODOS LOS TEXTOS EN ESPAÑOL):
{
  "turbidity_ntu": <número entre 0 y 150>,
  "confidence_score": <porcentaje 0-100 basado en calidad de imagen y claridad de indicadores>,
  "visual_observations": {
    "clarity": "<descripción breve de la claridad del agua en español>",
    "color_tint": "<color presente: transparente/amarillento/marrón/verdoso/etc en español>",
    "visible_particles": "<ninguna/pocas/moderadas/muchas/abundantes en español>",
    "light_transmission": "<excelente/buena/aceptable/pobre en español>"
  },
  "quality_indicators": {
    "suspended_solids": "<bajo/medio/alto en español>",
    "sediment_presence": "<ninguno/mínimo/moderado/abundante en español>",
    "organic_matter": "<no visible/posiblemente presente/claramente presente en español>"
  },
  "treatment_recommendations": [
    "<acción específica 1 en español, máximo 100 caracteres>",
    "<acción específica 2 en español, máximo 100 caracteres>"
  ],
  "potential_causes": [
    "<causa probable 1 en español, máximo 80 caracteres>",
    "<causa probable 2 en español, máximo 80 caracteres>"
  ],
  "image_quality_notes": "<notas sobre iluminación, enfoque o calidad de imagen en español, máximo 150 caracteres>"
}

IMPORTANTE: Todos los textos descriptivos deben estar en ESPAÑOL y ser concisos para mostrarse correctamente en la interfaz.
"""

# ---------------------------------------------------------
# Función principal mejorada
# ---------------------------------------------------------
def analyze_water_turbidity(image_bytes):
    """
    Analiza la turbidez del agua usando OpenAI GPT-4 Vision API.
    
    Args:
        image_bytes: Bytes de la imagen subida
        
    Returns:
        dict: Análisis completo de turbidez con métricas detalladas
    """
    try:
        # Verificar que existe API key
        if not os.getenv('OPENAI_API_KEY'):
            return {
                'error': True,
                'message': 'Error: OPENAI_API_KEY no configurada. Por favor añádela a tu archivo .env',
                'ntu': None
            }
        
        # Convertir imagen a base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Llamada a OpenAI Vision API
        response = client.chat.completions.create(
            #model="gpt-4o",  # o "gpt-4-vision-preview" según disponibilidad
            model="gpt-5.2",
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            #max_tokens=800,
            temperature=0.3  # Baja temperatura para respuestas más consistentes
        )
        
        # Extraer respuesta
        raw_output = response.choices[0].message.content.strip()
        
        # Limpiar markdown si existe
        if raw_output.startswith('```json'):
            raw_output = raw_output.replace('```json', '').replace('```', '').strip()
        elif raw_output.startswith('```'):
            raw_output = raw_output.replace('```', '').strip()
        
        # Parsear JSON
        ai_analysis = json.loads(raw_output)
        
        # Extraer valores principales
        ntu_value = float(ai_analysis.get('turbidity_ntu', 0))
        confidence = int(ai_analysis.get('confidence_score', 75))
        
        # Clasificación basada en NTU
        if ntu_value < 1:
            classification = "Excelente"
            status = "safe"
        elif ntu_value < 5:
            classification = "Muy Buena"
            status = "safe"
        elif ntu_value < 10:
            classification = "Buena"
            status = "acceptable"
        elif ntu_value < 25:
            classification = "Aceptable"
            status = "acceptable"
        elif ntu_value < 50:
            classification = "Deficiente"
            status = "poor"
        else:
            classification = "Muy Turbia"
            status = "poor"
        
        # Construir perfil visual para compatibilidad con UI
        visual_obs = ai_analysis.get('visual_observations', {})
        
        # Función auxiliar para truncar texto
        def truncate_text(text, max_length=50):
            """Trunca texto largo para mejor visualización en UI"""
            if not text or text == 'No disponible':
                return text
            text = str(text).strip()
            return text if len(text) <= max_length else text[:max_length-3] + '...'
        
        color_profile = {
            'clarity': truncate_text(visual_obs.get('clarity', 'No disponible')),
            'color_tint': truncate_text(visual_obs.get('color_tint', 'No disponible'), 30),
            'visible_particles': truncate_text(visual_obs.get('visible_particles', 'No disponible'), 20),
            'light_transmission': truncate_text(visual_obs.get('light_transmission', 'No disponible'), 20)
        }
        
        # Construir recomendación
        recommendations = ai_analysis.get('treatment_recommendations', [])
        recommendation_text = get_recommendation(ntu_value)
        if recommendations:
            # Limitar a 2 recomendaciones principales
            top_recommendations = recommendations[:2]
            recommendation_text += "\n\n🔧 Recomendaciones AI:\n" + "\n".join(f"• {r}" for r in top_recommendations)
        
        # Procesar quality indicators con valores truncados
        quality_indicators = ai_analysis.get('quality_indicators', {})
        quality_indicators_clean = {
            'suspended_solids': truncate_text(quality_indicators.get('suspended_solids', 'N/A'), 20),
            'sediment_presence': truncate_text(quality_indicators.get('sediment_presence', 'N/A'), 20),
            'organic_matter': truncate_text(quality_indicators.get('organic_matter', 'N/A'), 30)
        }
        
        # Procesar causas potenciales
        potential_causes = ai_analysis.get('potential_causes', [])
        potential_causes_clean = [truncate_text(cause, 100) for cause in potential_causes[:3]]
        
        # Notas de calidad de imagen
        image_notes = truncate_text(ai_analysis.get('image_quality_notes', ''), 200)
        
        return {
            'ntu': round(ntu_value, 2),
            'classification': classification,
            'status': status,
            'confidence': confidence,
            'color_profile': color_profile,
            'recommendation': recommendation_text,
            'meets_who_standards': ntu_value < 5,
            'ai_insights': {
                'quality_indicators': quality_indicators_clean,
                'potential_causes': potential_causes_clean,
                'image_quality_notes': image_notes
            },
            'powered_by': 'OpenAI GPT-4 Vision'
        }
        
    except json.JSONDecodeError as e:
        # Mejor manejo de errores JSON con preview de respuesta
        preview = raw_output[:300] if len(raw_output) > 300 else raw_output
        return {
            'error': True,
            'message': f"❌ Error al interpretar respuesta de OpenAI.\n\nError: {str(e)}\n\nRespuesta recibida:\n{preview}",
            'ntu': None
        }
    except Exception as e:
        error_msg = str(e)
        # Mensajes de error más amigables
        if 'api_key' in error_msg.lower():
            friendly_msg = "🔑 Error de autenticación: Verifica tu OPENAI_API_KEY en el archivo .env"
        elif 'quota' in error_msg.lower() or 'insufficient' in error_msg.lower():
            friendly_msg = "💳 Cuota excedida: Tu cuenta de OpenAI necesita créditos"
        elif 'connection' in error_msg.lower() or 'timeout' in error_msg.lower():
            friendly_msg = "🌐 Error de conexión: Verifica tu conexión a internet"
        else:
            friendly_msg = f"⚠️ Error en análisis con OpenAI: {error_msg}"
        
        return {
            'error': True,
            'message': friendly_msg,
            'ntu': None
        }

# ---------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------
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
    elif ntu_value < 50:
        return "🚫 No apta para consumo. Requiere tratamiento intensivo."
    else:
        return "❌ Turbidez extremadamente alta. Tratamiento crítico requerido."

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
            {'range': '25-50 NTU', 'quality': 'Deficiente', 'description': 'No apta para consumo'},
            {'range': '>50 NTU', 'quality': 'Muy Turbia', 'description': 'Calidad crítica'}
        ]
    }
