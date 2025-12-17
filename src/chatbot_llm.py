"""
Módulo de Chatbot LLM para el Sistema de Predicción de Calidad de Agua
Soporta múltiples proveedores: OpenAI, Google Gemini, Anthropic, OpenRouter
"""

import os
import json
import requests
from typing import List, Dict, Tuple
import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Importaciones condicionales para cada proveedor
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from google import genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# OpenRouter siempre está disponible (usa requests)
OPENROUTER_AVAILABLE = True


class ChatbotLLM:
    """
    Clase principal del chatbot que maneja múltiples proveedores de LLM
    """
    
    def __init__(self, provider: str = "openai"):
        """
        Inicializa el chatbot con el proveedor especificado
        
        Args:
            provider: 'openai', 'google', 'anthropic', o 'openrouter'
        
        Note:
            Las API keys se cargan automáticamente desde variables de entorno (.env):
            - OPENAI_API_KEY para OpenAI
            - GOOGLE_API_KEY para Google Gemini
            - ANTHROPIC_API_KEY para Anthropic
            - OPENROUTER_API_KEY para OpenRouter
        """
        self.provider = provider.lower()
        
        # Cargar API key desde variables de entorno según el proveedor
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY"
        }
        
        env_var = env_key_map.get(self.provider)
        if env_var:
            self.api_key = os.getenv(env_var)
            if not self.api_key:
                raise ValueError(f"Variable de entorno '{env_var}' no encontrada. Configúrala en tu archivo .env")
        else:
            raise ValueError(f"Proveedor '{self.provider}' no soportado")
        
        self.conversation_history = []
        
        # Contexto del sistema sobre el proyecto
        self.system_context = """
Eres un asistente experto en calidad de agua y análisis de potabilidad. 
Trabajas en el Sistema de Predicción de Calidad de Agua (SIPCA) para plantas de tratamiento.

Tu conocimiento incluye:
- Parámetros físico-químicos del agua: pH, dureza, sólidos disueltos, cloraminas, sulfatos, conductividad, carbono orgánico, trihalometanos y turbidez
- Normativas de calidad de agua potable (OMS, EPA)
- Interpretación de resultados de análisis de agua
- Machine Learning aplicado a predicción de potabilidad

Debes:
1. Responder de forma clara y profesional
2. Explicar conceptos técnicos de manera accesible
3. Proporcionar recomendaciones basadas en evidencia
4. Alertar sobre valores fuera de norma
5. Ser conciso pero completo

Rangos seguros de referencia:
- pH: 6.5 - 8.5
- Dureza: 50 - 300 mg/L
- Sólidos: < 500 ppm (TDS)
- Cloraminas: 0.2 - 4 ppm
- Sulfatos: < 250 mg/L
- Conductividad: 50 - 800 µS/cm
- Trihalometanos: < 80 ppb
- Turbidez: < 5 NTU
"""
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializa el cliente del proveedor seleccionado"""
        if self.provider == "openai" and OPENAI_AVAILABLE:
            openai.api_key = self.api_key
            self.client = openai.OpenAI(api_key=self.api_key)
            
        elif self.provider == "google" and GOOGLE_AVAILABLE:
            self.client = genai.Client(api_key=self.api_key)
            
        elif self.provider == "anthropic" and ANTHROPIC_AVAILABLE:
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
        elif self.provider == "openrouter" and OPENROUTER_AVAILABLE:
            # OpenRouter no necesita cliente, usa requests directamente
            self.client = None  # Usaremos requests en get_response_openrouter
            
        else:
            raise ValueError(f"Proveedor '{self.provider}' no disponible o no instalado")
    
    def add_message(self, role: str, content: str):
        """Añade un mensaje al historial de conversación"""
        self.conversation_history.append({
            "role": role,
            "content": content
        })
    
    def clear_history(self):
        """Limpia el historial de conversación"""
        self.conversation_history = []
    
    def get_response_openai(self, user_message: str) -> str:
        """Obtiene respuesta usando OpenAI GPT"""
        try:
            # Construir mensajes
            messages = [
                {"role": "system", "content": self.system_context}
            ]
            messages.extend(self.conversation_history)
            messages.append({"role": "user", "content": user_message})
            
            # Llamada a la API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Puedes cambiar a "gpt-4" si tienes acceso
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error OpenAI: {str(e)}"
    
    def get_response_google(self, user_message: str) -> str:
        """Obtiene respuesta usando Google Gemini"""
        try:
            # Construir el prompt completo
            full_prompt = f"{self.system_context}\n\n"
            
            # Añadir historial
            for msg in self.conversation_history:
                role = "Usuario" if msg["role"] == "user" else "Asistente"
                full_prompt += f"{role}: {msg['content']}\n"
            
            full_prompt += f"Usuario: {user_message}\nAsistente:"
            
            # Llamada a la API
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=full_prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"Error Gemini: {str(e)}"
    
    def get_response_anthropic(self, user_message: str) -> str:
        """Obtiene respuesta usando Anthropic Claude"""
        try:
            # Construir mensajes
            messages = []
            for msg in self.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            messages.append({"role": "user", "content": user_message})
            
            # Llamada a la API
            response = self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                system=self.system_context,
                messages=messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error Anthropic: {str(e)}"
    
    def get_response_openrouter(self, user_message: str) -> str:
        """Obtiene respuesta usando OpenRouter con fallback automático de modelos"""
        import time
        
        # Lista de modelos gratuitos en orden de preferencia
        # Se intentarán en orden hasta encontrar uno que funcione
        FALLBACK_MODELS = [
            "cognitivecomputations/dolphin-mistral-24b-venice-edition:free",
            "nousresearch/hermes-3-llama-3.1-405b:free",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2-7b-instruct:free",
            "microsoft/phi-3-mini-128k-instruct:free",
            "mistralai/mistral-small-3.1-24b-instruct:free",
        ]
        
        try:
            # Construir mensajes
            messages = [
                {"role": "system", "content": self.system_context}
            ]
            
            # Añadir historial
            for msg in self.conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            messages.append({"role": "user", "content": user_message})
            
            # Intentar con cada modelo en la lista
            last_error = None
            for model_name in FALLBACK_MODELS:
                try:
                    # Llamada a OpenRouter API (según documentación oficial)
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                            "HTTP-Referer": "https://sipca-water-quality.app",
                            "X-Title": "SIPCA - Water Quality Prediction",
                        },
                        data=json.dumps({
                            "model": model_name,
                            "messages": messages
                        }),
                        timeout=30
                    )
                    
                    # Si es exitoso, retornar
                    if response.status_code == 200:
                        result = response.json()
                        return result['choices'][0]['message']['content']
                    
                    # Si es 404, probar siguiente modelo
                    if response.status_code == 404:
                        continue
                    
                    # Si es rate limit (429), esperar y reintentar con el mismo modelo
                    if response.status_code == 429:
                        retry_after = response.headers.get('Retry-After', '2')
                        try:
                            delay = min(int(retry_after), 10)
                        except:
                            delay = 2
                        time.sleep(delay)
                        continue
                    
                    # Otros errores, guardar y probar siguiente modelo
                    last_error = f"HTTP {response.status_code}"
                    continue
                    
                except requests.exceptions.Timeout:
                    # Timeout, probar siguiente modelo
                    last_error = "Timeout"
                    continue
                except requests.exceptions.RequestException as e:
                    # Error de conexión, probar siguiente modelo
                    last_error = str(e)
                    continue
            
            # Si ningún modelo funcionó, retornar mensaje amigable
            if last_error and "429" in str(last_error):
                return "⏳ Has excedido el límite de solicitudes. Espera unos minutos e intenta de nuevo."
            elif last_error and "404" in str(last_error):
                return "🔍 Los modelos no están disponibles en este momento. Intenta más tarde."
            else:
                return "😔 El servicio no está disponible en este momento. Intenta más tarde."
            
        except Exception:
            return "😔 El servicio no está disponible en este momento. Intenta más tarde."
    
    def chat(self, user_message: str) -> str:
        """
        Método principal para chatear
        
        Args:
            user_message: Mensaje del usuario
            
        Returns:
            Respuesta del LLM
        """
        # Añadir mensaje del usuario al historial
        self.add_message("user", user_message)
        
        # Obtener respuesta según el proveedor
        if self.provider == "openai":
            response = self.get_response_openai(user_message)
        elif self.provider == "google":
            response = self.get_response_google(user_message)
        elif self.provider == "anthropic":
            response = self.get_response_anthropic(user_message)
        elif self.provider == "openrouter":
            response = self.get_response_openrouter(user_message)
        else:
            response = "Proveedor no soportado"
        
        # Añadir respuesta al historial
        self.add_message("assistant", response)
        
        return response


def get_available_providers() -> List[str]:
    """Retorna lista de proveedores disponibles"""
    providers = []
    if OPENROUTER_AVAILABLE:
        providers.append("OpenRouter (Múltiples modelos GRATIS)")
    if OPENAI_AVAILABLE:
        providers.append("OpenAI (GPT)")
    if GOOGLE_AVAILABLE:
        providers.append("Google (Gemini)")
    if ANTHROPIC_AVAILABLE:
        providers.append("Anthropic (Claude)")
    return providers


def create_chatbot_widget():
    """
    Crea un widget de chatbot con diseño moderno y limpio
    Estilo similar a chatbots web profesionales
    """
    
    # Inicializar estado de sesión
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None
    
    if "chat_expanded" not in st.session_state:
        st.session_state.chat_expanded = False
    
    if "last_error" not in st.session_state:
        st.session_state.last_error = None
    
    # CSS moderno y limpio
    st.markdown("""
    <style>
    /* Contenedor principal del widget */
    .chat-widget-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        font-family: 'Space Grotesk', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Panel del chat */
    .chat-panel {
        width: 400px;
        max-height: 600px;
        background: white;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
        overflow: hidden;
        animation: slideUp 0.3s ease-out;
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Header del chat */
    .chat-header {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 24px;
        border-bottom: 1px solid #E3F2FD;
    }
    
    .chat-welcome {
        font-size: 28px;
        font-weight: 600;
        color: #1565C0;
        margin: 0 0 8px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .chat-subtitle {
        font-size: 18px;
        color: #424242;
        margin: 0;
        font-weight: 400;
    }
    
    /* Input del chat */
    .chat-input-section {
        padding: 16px 20px;
        background: white;
        border-bottom: 1px solid #E0E0E0;
    }
    
    /* Sección de inicio */
    .chat-start-section {
        padding: 20px;
        background: #FAFAFA;
        border-bottom: 1px solid #E0E0E0;
    }
    
    .chat-start-button {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        background: white;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .chat-start-button:hover {
        background: #F5F5F5;
        border-color: #1976D2;
    }
    
    .chat-disclaimer {
        font-size: 11px;
        color: #757575;
        margin-top: 8px;
        line-height: 1.4;
    }
    
    /* Bookmarks */
    .chat-bookmarks {
        padding: 16px 20px;
    }
    
    .bookmarks-title {
        font-size: 13px;
        font-weight: 600;
        color: #616161;
        margin-bottom: 12px;
    }
    
    .bookmark-item {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        margin-bottom: 6px;
        border-radius: 6px;
        cursor: pointer;
        transition: background 0.2s;
        font-size: 14px;
        color: #424242;
    }
    
    .bookmark-item:hover {
        background: #F5F5F5;
    }
    
    /* Footer */
    .chat-footer {
        padding: 12px 20px;
        text-align: center;
        font-size: 11px;
        color: #9E9E9E;
        border-top: 1px solid #E0E0E0;
    }
    
    /* Mensajes del chat */
    .stChatMessage {
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 12px;
    }
    
    /* Botón flotante */
    .chat-float-btn {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%);
        box-shadow: 0 4px 16px rgba(25, 118, 210, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 4px 16px rgba(25, 118, 210, 0.4);
        }
        50% {
            box-shadow: 0 4px 24px rgba(25, 118, 210, 0.6);
        }
    }
    
    .chat-float-btn:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 24px rgba(25, 118, 210, 0.5);
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .chat-panel {
            width: calc(100vw - 40px);
            max-height: calc(100vh - 100px);
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Configuración en sidebar
    with st.sidebar.expander("🤖 Configurar Asistente IA", expanded=False):
        st.markdown("#### Conexión del Chatbot")
        
        available = get_available_providers()
        if not available:
            st.warning("⚠️ Instala un proveedor LLM")
            st.code("pip install google-generativeai", language="bash")
            st.caption("Google Gemini es **GRATIS** 💳")
            return
        
        provider_map = {
            "OpenRouter (Múltiples modelos GRATIS)": "openrouter",
            "OpenAI (GPT)": "openai",
            "Google (Gemini)": "google",
            "Anthropic (Claude)": "anthropic"
        }
        
        env_var_map = {
            "openrouter": "OPENROUTER_API_KEY",
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"
        }
        
        selected_provider = st.selectbox(
            "Proveedor",
            available,
            key="llm_provider",
            help="💡 OpenRouter da acceso GRATIS a Gemini, Llama, y más"
        )
        
        # Espaciado
        st.markdown("")
        
        # Verificar qué variable de entorno se necesita
        provider_code = provider_map[selected_provider]
        required_env_var = env_var_map[provider_code]
        
        # Verificar si la variable de entorno está configurada
        is_configured = os.getenv(required_env_var) is not None
        
        if is_configured:
            st.success(f"✅ {required_env_var} configurada")
        else:
            st.warning(f"⚠️ Configura {required_env_var} en tu archivo .env")
            st.caption("🔗 OpenRouter: openrouter.ai/keys | Gemini: makersuite.google.com")
        
        # Espaciado antes del botón
        st.markdown("")
        
        # Botón de conectar (ocupa todo el ancho)
        if st.button("🔌 Conectar", use_container_width=True, disabled=not is_configured):
            try:
                st.session_state.chatbot = ChatbotLLM(provider=provider_code)
                st.session_state.last_error = None  # Limpiar errores previos
                st.success("✅ Conectado!")
                st.balloons()
            except Exception as e:
                error_msg = str(e)
                st.session_state.last_error = error_msg
                st.error(f"❌ Error de conexión")
        
        st.divider()
        
        # Estado de conexión
        if st.session_state.chatbot:
            st.success(f"🟢 **Conectado:** {selected_provider}")
            st.caption(f"💬 {len(st.session_state.chat_messages)} mensajes")
        else:
            st.info("🔴 **Desconectado**")
        
        # Mostrar último error técnico (solo para debugging)
        if st.session_state.last_error:
            with st.expander("⚠️ Último error técnico", expanded=False):
                st.code(st.session_state.last_error, language="text")
                if st.button("🗑️ Limpiar error", key="clear_error"):
                    st.session_state.last_error = None
                    st.rerun()
    
    # CSS para posicionar el widget en la esquina inferior derecha (fijo)
    st.markdown("""
    <style>
    /* Forzar SOLO el popover del chatbot a estar en la esquina inferior derecha */
    /* NO afectar a selectbox ni otros popovers */
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) {
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        z-index: 9999 !important;
    }
    
    /* Forzar SOLO el contenido del popover del chatbot */
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) [data-baseweb="popover"],
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) > div:not([data-baseweb="select"]) {
        position: fixed !important;
        bottom: 90px !important;
        right: 20px !important;
        left: auto !important;
        top: auto !important;
        transform: none !important;
        margin: 0 !important;
    }
    
    /* Ajustar el ancho SOLO de la ventana del chat */
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) [data-baseweb="popover"] > div,
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) > div > div:not([data-baseweb="select"]) {
        width: 400px !important;
        max-width: 90vw !important;
    }
    
    /* NO afectar los selectbox - dejarlos con su comportamiento normal */
    [data-baseweb="select"],
    [data-baseweb="popover"]:has([role="listbox"]) {
        position: absolute !important;
        bottom: auto !important;
        left: auto !important;
        transform: initial !important;
    }
    
    /* Estilo del botón flotante del chat */
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) button {
        width: 60px !important;
        height: 60px !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #1976D2 0%, #42A5F5 100%) !important;
        box-shadow: 0 4px 16px rgba(25, 118, 210, 0.4) !important;
        border: none !important;
        font-size: 24px !important;
        animation: pulse 2s infinite !important;
    }
    
    @keyframes pulse {
        0%, 100% {
            box-shadow: 0 4px 16px rgba(25, 118, 210, 0.4);
        }
        50% {
            box-shadow: 0 4px 24px rgba(25, 118, 210, 0.6);
        }
    }
    
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) button:hover {
        transform: scale(1.05) !important;
        box-shadow: 0 6px 24px rgba(25, 118, 210, 0.5) !important;
    }
    
        /* Icono del chatbot flotante */
    [data-testid="stPopover"]:not([data-testid="stSelectbox"]) button::before {
        content: "smart_toy";
        font-family: 'Material Symbols Outlined';
        font-size: 28px;
        font-weight: normal;
        font-variation-settings: 'FILL' 1;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Widget del chatbot (sin columnas, se posiciona con CSS)
    with st.popover("", help="Asistente de Calidad de Agua"):
        # Verificar conexión
        if st.session_state.chatbot is None:
            # Pantalla de configuración
            st.markdown("""
            <div style="padding: 20px;">
                <p style="color: #757575; font-size: 14px; margin-bottom: 16px;">
                    Para empezar a chatear, configura tu asistente:
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👈 Ve a la barra lateral y expande **'🤖 Configurar Asistente IA'**")
            
            st.markdown("""
            <div style="padding: 20px; background: #F5F5F5; border-radius: 8px; margin: 16px;">
                <p style="font-size: 13px; font-weight: 600; margin-bottom: 8px;">Pasos rápidos:</p>
                <ol style="font-size: 12px; color: #616161; margin: 0; padding-left: 20px;">
                    <li>Selecciona un proveedor (ej: "OpenRouter")</li>
                    <li>Configura la variable de entorno en tu archivo .env</li>
                    <li>Reinicia la aplicación</li>
                    <li>Haz clic en "🔌 Conectar"</li>
                </ol>
                <p style="font-size: 11px; color: #757575; margin-top: 12px;">
                    💡 Las API keys se cargan automáticamente desde el archivo .env
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Bookmarks
            st.markdown("""
            <div class="chat-bookmarks">
                <div class="bookmarks-title">Recursos útiles</div>
                <div class="bookmark-item">📚 Docs - Documentación completa</div>
                <div class="bookmark-item">💬 Community - Foro de ayuda</div>
                <div class="bookmark-item">🌐 Website - Sitio web oficial</div>
                <div class="bookmark-item">❓ Help Center - Centro de ayuda</div>
            </div>
            """, unsafe_allow_html=True)
            
            return
        
        # Chat activo
        st.success("🟢 Asistente conectado")
        
        # Input del chat
        st.markdown("""
        <div class="chat-input-section">
            <div style="display: flex; align-items: center; gap: 8px; color: #1976D2;">
                <span style="font-size: 20px;">💧</span>
                <span style="font-size: 13px; font-weight: 500;">Pregunta sobre calidad de agua</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Contenedor de mensajes
        chat_container = st.container(height=300)
        
        with chat_container:
            if len(st.session_state.chat_messages) == 0:
                # Mensaje de bienvenida
                st.markdown("""
                <div style="text-align: center; padding: 40px 20px; color: #757575;">
                    <div style="font-size: 48px; margin-bottom: 16px;">👋</div>
                    <h4 style="margin: 0 0 12px 0; color: #424242; font-weight: 500;">
                        ¡Hola! Soy tu asistente de agua
                    </h4>
                    <p style="margin: 0; font-size: 13px; color: #757575;">
                        Puedo ayudarte con:
                    </p>
                    <div style="text-align: left; display: inline-block; margin-top: 16px; font-size: 12px; color: #616161;">
                        • Parámetros de calidad (pH, dureza, TDS...)<br>
                        • Normativas OMS y EPA<br>
                        • Interpretación de análisis<br>
                        • Recomendaciones de tratamiento
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Mostrar historial
                for message in st.session_state.chat_messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
        
        # Input
        if prompt := st.chat_input("Escribe tu pregunta aquí...", key="chat_input_widget"):
            st.session_state.chat_messages.append({
                "role": "user",
                "content": prompt
            })
            
            with st.spinner("💭 Pensando..."):
                try:
                    response = st.session_state.chatbot.chat(prompt)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.session_state.last_error = None  # Limpiar error si la respuesta fue exitosa
                    st.rerun()
                except Exception as e:
                    # Guardar error técnico para el sidebar
                    st.session_state.last_error = str(e)
                    
                    # Mostrar mensaje amigable en el chat
                    error_msg = "😔 **Lo siento, no pude procesar tu pregunta**\n\n"
                    
                    # Detectar tipo de error y dar mensaje específico
                    error_str = str(e).lower()
                    if "404" in error_str or "not found" in error_str:
                        error_msg += "🔍 El modelo no está disponible en este momento.\n\n💡 **Sugerencia**: Verifica que el modelo esté activo o prueba con otro proveedor."
                    elif "429" in error_str or "rate limit" in error_str or "quota" in error_str:
                        error_msg += "⏳ Has excedido el límite de solicitudes.\n\n💡 **Sugerencia**: Espera unos minutos e intenta de nuevo, o considera usar otro proveedor."
                    elif "401" in error_str or "unauthorized" in error_str or "api key" in error_str:
                        error_msg += "🔑 Problema con la API key.\n\n💡 **Sugerencia**: Verifica que tu API key sea válida y esté correctamente configurada en el archivo .env"
                    elif "timeout" in error_str:
                        error_msg += "⏱️ La solicitud tardó demasiado tiempo.\n\n💡 **Sugerencia**: Intenta de nuevo en unos momentos."
                    else:
                        error_msg += "❌ Ocurrió un error inesperado.\n\n💡 **Sugerencia**: Revisa la sección de configuración en el sidebar para más detalles técnicos."
                    
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
                    st.rerun()
        
        # Disclaimer
        st.markdown("""
        <div class="chat-disclaimer" style="padding: 12px 16px; text-align: center;">
            AI-generated responses may not always be accurate. Please verify important information.
        </div>
        """, unsafe_allow_html=True)
