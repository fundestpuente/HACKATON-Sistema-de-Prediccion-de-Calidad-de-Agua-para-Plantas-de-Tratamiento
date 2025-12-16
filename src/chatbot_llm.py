"""
Módulo de Chatbot LLM para el Sistema de Predicción de Calidad de Agua
Soporta múltiples proveedores: OpenAI, Google Gemini, Anthropic, OpenRouter
"""

import os
import json
import requests
from typing import List, Dict, Tuple
import streamlit as st

# Importaciones condicionales para cada proveedor
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
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
    
    def __init__(self, provider: str = "openai", api_key: str = None):
        """
        Inicializa el chatbot con el proveedor especificado
        
        Args:
            provider: 'openai', 'google', o 'anthropic'
            api_key: API key del proveedor
        """
        self.provider = provider.lower()
        self.api_key = api_key
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
            genai.configure(api_key=self.api_key)
            # Usar el nombre completo del modelo con prefijo 'models/'
            # Esto es compatible con todas las versiones de la API
            self.client = genai.GenerativeModel('models/gemini-pro')
            
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
            return f"Error al conectar con OpenAI: {str(e)}"
    
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
            response = self.client.generate_content(full_prompt)
            
            return response.text
            
        except Exception as e:
            return f"Error al conectar con Google Gemini: {str(e)}"
    
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
            return f"Error al conectar con Anthropic: {str(e)}"
    
    def get_response_openrouter(self, user_message: str) -> str:
        """Obtiene respuesta usando OpenRouter con backoff exponencial"""
        import time
        
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
            
            # Backoff exponencial (mejores prácticas de OpenRouter)
            max_retries = 3
            base_delay = 1  # 1 segundo inicial
            max_delay = 60  # máximo 60 segundos
            
            for attempt in range(max_retries):
                try:
                    # Llamada a OpenRouter API
                    response = requests.post(
                        url="https://openrouter.ai/api/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            # "HTTP-Referer": "https://sipca-water-quality.app",
                            # "X-Title": "SIPCA - Water Quality Prediction",
                            "Content-Type": "application/json"
                        },
                        json={
                            # Modelos gratuitos con mejores límites:
                            # "meta-llama/llama-3.1-8b-instruct:free" - 20 RPM (RECOMENDADO)
                            # "microsoft/phi-3-mini-128k-instruct:free" - 20 RPM
                            # "google/gemma-2-9b-it:free" - 20 RPM
                            "model": "meta-llama/llama-3.1-8b-instruct:free",
                            "messages": messages,
                            "max_tokens": 500,
                            "temperature": 0.7
                        },
                        timeout=30
                    )
                    
                    # Si es exitoso, retornar
                    if response.status_code == 200:
                        result = response.json()
                        return result['choices'][0]['message']['content']
                    
                    # Si es rate limit (429), aplicar backoff exponencial
                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            # Calcular delay con backoff exponencial
                            delay = min(base_delay * (2 ** attempt), max_delay)
                            
                            # Intentar obtener el tiempo de retry del header
                            retry_after = response.headers.get('Retry-After')
                            if retry_after:
                                try:
                                    delay = int(retry_after)
                                except:
                                    pass
                            
                            # Mostrar mensaje de espera (solo en desarrollo)
                            # print(f"Rate limit alcanzado. Esperando {delay}s antes de reintentar...")
                            time.sleep(delay)
                            continue
                        else:
                            return "⏳ **Límite de requests alcanzado**\n\nPor favor espera 1 minuto e intenta de nuevo.\n\n💡 **Tip**: Los modelos gratuitos tienen límites de 20 requests/minuto."
                    
                    # Otros errores HTTP
                    response.raise_for_status()
                
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429 and attempt < max_retries - 1:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        time.sleep(delay)
                        continue
                    raise
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                return "⏳ **Límite de requests alcanzado**\n\nEspera 1 minuto y vuelve a intentar.\n\n💡 **Tip**: Los modelos gratis tienen límites de 20 requests/minuto."
            elif e.response.status_code == 401:
                return "❌ **API Key inválida**\n\nVerifica tu API key en: https://openrouter.ai/keys"
            elif e.response.status_code == 402:
                return "💳 **Créditos insuficientes**\n\nEste modelo requiere créditos. Usa un modelo con sufijo ':free'"
            else:
                return f"❌ **Error HTTP {e.response.status_code}**\n\n{str(e)}"
        except requests.exceptions.Timeout:
            return "⏱️ **Timeout**\n\nLa respuesta tardó demasiado. Intenta de nuevo."
        except requests.exceptions.RequestException as e:
            return f"🌐 **Error de conexión**\n\n{str(e)}"
        except KeyError:
            return "❌ **Error en la respuesta**\n\nLa API retornó un formato inesperado."
        except Exception as e:
            return f"❌ **Error inesperado**\n\n{str(e)}"

    
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
        
        selected_provider = st.selectbox(
            "Proveedor",
            available,
            key="llm_provider",
            help="💡 OpenRouter da acceso GRATIS a Gemini, Llama, y más"
        )
        
        api_key = st.text_input(
            "API Key",
            type="password",
            key="llm_api_key",
            placeholder="Pega tu API key aquí",
            help="🔗 OpenRouter: openrouter.ai/keys | Gemini: makersuite.google.com"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔌 Conectar", use_container_width=True):
                if not api_key:
                    st.error("⚠️ Ingresa API Key")
                else:
                    try:
                        provider_code = provider_map[selected_provider]
                        st.session_state.chatbot = ChatbotLLM(
                            provider=provider_code,
                            api_key=api_key
                        )
                        st.success("✅ Conectado!")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ {str(e)[:80]}...")
        
        with col2:
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.chat_messages = []
                if st.session_state.chatbot:
                    st.session_state.chatbot.clear_history()
                st.success("🧹 Limpiado!")
        
        st.divider()
        if st.session_state.chatbot:
            st.success(f"🟢 **Conectado:** {selected_provider}")
            st.caption(f"💬 {len(st.session_state.chat_messages)} mensajes")
        else:
            st.info("🔴 **Desconectado**")
    
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
    </style>
    """, unsafe_allow_html=True)
    
    # Widget del chatbot (sin columnas, se posiciona con CSS)
    with st.popover("💬", help="Asistente de Calidad de Agua"):
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
                    <li>Selecciona "OpenRouter"</li>
                    <li>Añade OPENROUTER_API_KEY a tu .env</li>
                    <li>Haz clic en "🔌 Conectar"</li>
                </ol>
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
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)[:100]}...")
                    st.session_state.chat_messages.pop()
        
        # Disclaimer
        st.markdown("""
        <div class="chat-disclaimer" style="padding: 12px 16px; text-align: center;">
            AI-generated responses may not always be accurate. Please verify important information.
        </div>
        """, unsafe_allow_html=True)
