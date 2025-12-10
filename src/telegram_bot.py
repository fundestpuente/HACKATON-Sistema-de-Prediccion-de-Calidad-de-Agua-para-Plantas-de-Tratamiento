import os
import json
import requests
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler

# Cargar entorno
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Archivo compartido para guardar el ID (se guarda en la raiz del proyecto)
SHARED_FILE = "telegram_connection.json"

# ==========================================
# PARTE A: FUNCIÓN PARA ENVIAR (Uso en App.py)
# ==========================================
def send_telegram_alert(message, chat_id):
    """
    Envía un mensaje usando requests (síncrono) para no bloquear Streamlit.
    """
    if not TOKEN:
        return False, "No hay TOKEN en .env"
    
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True, "Enviado"
        return False, f"Error API: {response.text}"
    except Exception as e:
        return False, str(e)

# ==========================================
# PARTE B: LÓGICA DEL BOT ESCUCHA (Uso en Terminal)
# ==========================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    chat_id = update.effective_chat.id
    
    # Datos a guardar
    connection_data = {
        "chat_id": chat_id,
        "name": user.first_name,
        "username": user.username
    }
    
    # Guardar en JSON en la raiz
    try:
        with open(SHARED_FILE, "w") as f:
            json.dump(connection_data, f)
        
        print(f"✅ Nuevo usuario conectado: {user.first_name} (ID: {chat_id})")
        
        await context.bot.send_message(
            chat_id=chat_id, 
            text=f"👋 ¡Hola {user.first_name}!\n\nTe he registrado correctamente.\nAhora ve al Dashboard y presiona 'Sincronizar' para recibir las alertas aquí."
        )
    except Exception as e:
        print(f"Error escribiendo archivo: {e}")

def run_listener():
    """Inicia el bot para escuchar comandos"""
    if not TOKEN:
        print("❌ Error: No se encontró TELEGRAM_TOKEN en .env")
        return

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_command))
    
    print("🤖 Bot de Alertas ESCUCHANDO... (Presiona Ctrl+C para detener)")
    print(f"Esperando comando /start para guardar ID en '{SHARED_FILE}'...")
    app.run_polling()

# ==========================================
# PUNTO DE ENTRADA PRINCIPAL
# ==========================================
if __name__ == "__main__":
    run_listener()