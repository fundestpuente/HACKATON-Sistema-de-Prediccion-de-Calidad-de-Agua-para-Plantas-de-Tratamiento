import os
import json
import requests
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler
from datetime import datetime
# Cargar entorno
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Archivos compartidos (Base de datos simple)
CONNECTION_FILE = "telegram_connection.json" # Para guardar quién es el usuario
STATUS_FILE = "water_status.json"            # Para leer el último análisis del Dashboard
LOG_FILE = "maintenance_log.json"            # Archivo nuevo para guardar los reportes que vienen de Telegram

# --- DICCIONARIO DE DEFINICIONES ---
INFO_DICT = {
    "ph": "Medida de acidez/alcalinidad. Rango seguro: 6.5 - 8.5.",
    "turbidez": "Medida de la claridad del agua. Debe ser menor a 4.0 NTU.",
    "cloraminas": "Desinfectante usado para tratar el agua. Rango ideal: < 4 ppm.",
    "sulfatos": "Sales minerales. En exceso causan sabor amargo. Ideal < 250 mg/L.",
    "solidos": "Total de sólidos disueltos. Indica mineralización general."
}

# ==========================================
# PARTE A: FUNCIÓN PARA ENVIAR (Usada por app.py)
# ==========================================
def send_telegram_alert(message, chat_id):
    """Envía un mensaje de alerta (Unidireccional: App -> Telegram)"""
    if not TOKEN: return False, "No hay TOKEN"
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload)
        return (True, "Enviado") if response.status_code == 200 else (False, response.text)
    except Exception as e:
        return False, str(e)

# ==========================================
# PARTE B: COMANDOS DEL BOT (Bidireccional: Telegram <-> Usuario)
# ==========================================

# 1. Comando /start
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    # Guardamos al usuario
    data = {"chat_id": chat_id, "name": user.first_name}
    try:
        with open(CONNECTION_FILE, "w") as f:
            json.dump(data, f)
        msg = f"👋 ¡Hola {user.first_name}!\n\n✅ Conectado.\nVe al Dashboard y dale a 'Sincronizar' para recibir alertas.\n\nPrueba: /status , /ayuda o /info"
    except Exception as e:
        msg = f"Error guardando conexión: {e}"
        
    await context.bot.send_message(chat_id=chat_id, text=msg)

# 2. Comando /status (TAREA B)
async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lee el archivo que generó el Dashboard y responde al usuario"""
    try:
        with open(STATUS_FILE, "r") as f:
            data = json.load(f)
            
        # Determinar icono según potabilidad
        icon = "🟢" if data['prediction'] == "POTABLE" else "🔴"
        
        msg = (
            f"{icon} *ESTADO ACTUAL DEL SISTEMA*\n\n"
            f"💧 *Calidad:* {data['prediction']}\n"
            f"🧪 *pH Registrado:* {data['ph']:.2f}\n"
            f"📊 *Confianza IA:* {data['confidence']:.1f}%\n"
            f"🕒 *Último análisis:* {data.get('timestamp', 'Reciente')}"
        )
    except FileNotFoundError:
        msg = "🤷‍♂️ *No hay datos recientes.*\nEjecuta un análisis en el Dashboard primero."
    except Exception as e:
        msg = f"❌ Error leyendo estado: {str(e)}"

    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode="Markdown")

# 3. Comando /ayuda (TAREA C)
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Envía guía de actuación"""
    msg = (
        "🆘 *GUÍA DE ACCIÓN: AGUA NO POTABLE*\n\n"
        "1. 🚫 *DETENER:* Pare el bombeo inmediatamente.\n"
        "2. 🧪 *CORREGIR pH:* Si es < 6.5 añada base; si es > 8.5 añada ácido.\n"
        "3. ⚙️ *FILTROS:* Verifique saturación de filtros de carbón.\n"
        "4. 📞 *SOPORTE:* Contacte al ing. de planta si la turbidez es alta."
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=msg, parse_mode="Markdown")

# 4. NUEVO: Comando /info [parametro]
async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Explica un parámetro técnico. Uso: /info ph"""
    # Verificamos si el usuario escribió algo después de /info
    if not context.args:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, 
            text="ℹ️ Uso: `/info ph` o `/info turbidez`\nOpciones: ph, turbidez, cloraminas, sulfatos, solidos."
        )
        return

    param = context.args[0].lower() # Tomamos la primera palabra
    definition = INFO_DICT.get(param, "❌ Parámetro no encontrado. Prueba: ph, turbidez, cloraminas...")
    
    await context.bot.send_message(chat_id=update.effective_chat.id, text=definition)
# ==========================================
# LISTENER PRINCIPAL
# ==========================================
def run_listener():
    if not TOKEN:
        print("❌ Error: No Token")
        return

    app = ApplicationBuilder().token(TOKEN).build()
    
    # Registro de Comandos
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command)) # <--- Nuevo
    app.add_handler(CommandHandler("ayuda", help_command))    # <--- Nuevo
    app.add_handler(CommandHandler("info", info_command))
    print("🤖 Bot Inteligente ESCUCHANDO... (Comandos: /start, /status, /ayuda)")
    
    # stop_signals=None es vital para que funcione con Streamlit en hilos
    app.run_polling(stop_signals=None)

if __name__ == "__main__":
    run_listener()