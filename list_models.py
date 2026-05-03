import os
import requests
from dotenv import load_dotenv

# Cargar tu GOOGLE_API_KEY desde el archivo .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("❌ ERROR: No se encontró GOOGLE_API_KEY en tu .env")
    exit()

print("📡 Conectando a los servidores de Google (v1beta)...")
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
data = response.json()

if "models" in data:
    print("\n🟢 MODELOS DISPONIBLES PARA GENERAR TEXTO:\n" + "-"*40)
    for m in data["models"]:
        # Filtramos solo los que sirven para generar contenido (nuestro caso)
        if 'generateContent' in m.get('supportedGenerationMethods',[]):
            # Limpiamos el nombre para que veas exactamente qué poner en PydanticAI
            nombre_limpio = m['name'].replace('models/', '')
            print(f"👉 {nombre_limpio}")
    print("-"*40)
else:
    print("❌ Error en la respuesta:", data)