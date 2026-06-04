import os
import sys

# Asegurar que el directorio raíz del proyecto esté en el PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Agregar dinámicamente site-packages global para resolver grandalf si no está en el venv
global_site_packages = r"C:\Users\felix tremigual\AppData\Local\Programs\Python\Python313\Lib\site-packages"
if os.path.exists(global_site_packages):
    sys.path.append(global_site_packages)

try:
    from core.graph import katalog_agent
    print("\n" + "="*50)
    print("      DIAGRAMA DE FLUJO DE KATALOG-BRAIN (LangGraph)")
    print("="*50 + "\n")
    # Imprime el grafo en formato ASCII en la terminal
    print(katalog_agent.get_graph().draw_ascii())
    print("\n" + "="*50 + "\n")
except Exception as e:
    print(f"❌ Error al visualizar el grafo: {e}", file=sys.stderr)
    sys.exit(1)
