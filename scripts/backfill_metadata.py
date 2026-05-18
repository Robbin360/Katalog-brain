import os
import sys
import asyncio

# Auto-registro de ruta: permite importar módulos desde la raíz del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from supabase import create_client, Client
from agents.reclassifier_agent import classify_with_smart_fallback

load_dotenv()

# Inicialización de Supabase
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

async def process_rows():
    print("BACKFILL: Re-clasificando metadatos de Knowledge Base")
    print("============================================================")

    # 1. Buscamos los registros que NO tienen el metadato nuevo (buyer_archetype)
    # Usamos .is_('metadata->>buyer_archetype', 'null') para detectar los pendientes
    res = supabase.table('knowledge_base').select('id, content, metadata').execute()
    
    # Filtramos en Python para ser más precisos con JSONB
    rows_to_process = [
        row for row in res.data 
        if not row.get('metadata') or 'buyer_archetype' not in row['metadata']
    ]

    if not rows_to_process:
        print("✅ [INFO] No hay registros pendientes de re-clasificación.")
        return

    print(f"[INFO] {len(rows_to_process)} registros pendientes encontrados.\n")

    for row in rows_to_process:
        row_id = row['id']
        content = row['content']
        old_metadata = row['metadata'] or {}

        print(f"[PROC] ID {row_id} clasificando...", end=" ", flush=True)

        try:
            # 1. Clasifica con cascada de inteligencia (DeepSeek -> Qwen3-32B)
            result = await classify_with_smart_fallback(content)

            # 2. Extraccion blindada (compatible con PydanticAI v1/v2)
            new_data = getattr(result, 'data', getattr(result, 'output', result))

            # 3. Conversion segura a diccionario
            new_dict = new_data.model_dump() if hasattr(new_data, 'model_dump') else dict(new_data)

            # 4. Fusion de metadatos (preserva lo antiguo, anade lo nuevo)
            enriched_metadata = {**old_metadata, **new_dict}
            
            # 5. Actualizamos en Supabase
            supabase.table('knowledge_base').update({
                "metadata": enriched_metadata
            }).eq('id', row_id).execute()

            print("✅ ÉXITO")
            
            # ⏳ Pausa de seguridad de 1 segundo para no agotar la cuota de Google
            await asyncio.sleep(1)

        except Exception as e:
            print(f"❌ ERROR: {str(e)}")

    print("\n============================================================")
    print("🏁 [DONE] Proceso de Backfill finalizado.")

if __name__ == "__main__":
    # Arrancamos el bucle de forma correcta
    asyncio.run(process_rows())
