import os
import asyncio  # <-- Añadido para pausar el tiempo
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks  # <-- Añadido BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
from typing import Union, Optional
from supabase import create_client, Client

# Importamos el Grafo pesado y el Agente rápido
from core.graph import katalog_agent
from core.worker import auto_pilot_patrol
from agents.inspector_agent import inspector_agent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("🚀 [Servidor] Iniciando Katalog AI Brain...")
    auto_pilot_task = asyncio.create_task(auto_pilot_patrol())
    app.state.auto_pilot_task = auto_pilot_task
    print("🤖 [Servidor] Auto-Pilot Worker encendido.")

    try:
        yield
    finally:
        print("⏳ [Servidor] Apagando Auto-Pilot Worker...")
        auto_pilot_task.cancel()
        try:
            await auto_pilot_task
        except asyncio.CancelledError:
            print("✅ [Servidor] Auto-Pilot Worker detenido correctamente.")

# Inicializamos la API
app = FastAPI(
    title="Katalog AI Brain", 
    description="The Revenue Optimizer API - Powered by LangGraph & Gemini 3.1", 
    version="2.1",
    lifespan=lifespan
)

# 🛡️ CORS (Configurado para permitir comunicación con el Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción cambiaremos esto por tu dominio de Vercel
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializamos Supabase Globalmente para usarlo en el Triaje
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)


async def run_sync_io(callable_obj):
    return await asyncio.to_thread(callable_obj)

# --- MODELOS DE DATOS ---
class OptimizeRequest(BaseModel):
    product_id: Union[str, int] = Field(
        ..., 
        validation_alias=AliasChoices("product_id", "productId")
    )


# ==========================================
# ⚡ EL MOTOR DE TRIAJE (Trabajador en las sombras)
# ==========================================
async def process_triage_queue():
    """Esta función corre en el fondo sin bloquear la web"""
    print("🚨[TRIAJE BACKGROUND] Iniciando escaneo masivo de catálogo...")
    
    try:
        # 1. Buscar hasta 10 productos que no han sido auditados
        res = await run_sync_io(
            lambda: supabase.table('shopify_products')
            .select('id, current_title, current_body_html')
            .eq('audit_status', 'PENDING_AUDIT')
            .limit(10)
            .execute()
        )
        products = res.data
        
        if not products:
            print("✅ [TRIAJE BACKGROUND] Nada que auditar.")
            return

        print(f"📦 Procesando {len(products)} productos a velocidad de Radar (Max 4 por minuto)...")

        # 2. Auditar cada producto controlando la velocidad
        for prod in products:
            prompt = f"Title: {prod.get('current_title')}\nDescription: {prod.get('current_body_html')}"
            
            print(f"👀 Auditando: {prod.get('current_title')[:30]}...")
            
            # Llamamos al Inspector Flash
            result = await inspector_agent.run(prompt)
            # Extracción blindada de Pydantic
            audit_data = getattr(result, 'data', getattr(result, 'output', result))
            
            # 3. Guardar el veredicto en Supabase al instante
            await run_sync_io(lambda: supabase.table('shopify_products').update({
                'seo_score_initial': audit_data.score,
                'audit_score': audit_data.score,
                'error_log': audit_data.reason, 
                'audit_status': 'NEEDS_OPTIMIZATION' if audit_data.score < 90 else 'READY_TO_PUBLISH'
            }).eq('id', prod['id']).execute())
            
            print(f"✅ Score: {audit_data.score}/100 | Motivo: {audit_data.reason}")
            
            # 🛑 EL FRENO DE MANO (Rate Limit Bypass)
            # Pausamos 15 segundos para no enfadar a Google (Límite: 5 por minuto)
            print("⏳ Enfriando motor por 15 segundos para evitar ban de API...")
            await asyncio.sleep(15)

        print("🏁 [TRIAJE BACKGROUND] Lote terminado exitosamente.")

    except Exception as e:
        print(f"❌ [TRIAJE BACKGROUND] Error Fatal: {str(e)}")


# ==========================================
# ⚡ ENDPOINT 1: DISPARADOR DEL TRIAJE
# ==========================================
@app.post("/api/triage")
async def run_triage_scan(background_tasks: BackgroundTasks):
    # En lugar de hacer esperar al usuario, le pasamos el trabajo al empleado de fondo
    background_tasks.add_task(process_triage_queue)
    
    # Respondemos en 0.01 segundos
    return {
        "status": "ACCEPTED", 
        "message": "Triaje iniciado en segundo plano. Escaneando a 4 productos por minuto."
    }


# ==========================================
# 🧠 ENDPOINT 2: CIRUGÍA MAYOR (LANGGRAPH)
# ==========================================
@app.post("/api/optimize")
async def optimize_product(request: OptimizeRequest):
    # Convertimos siempre a string para que LangGraph y Supabase no tengan errores de tipos
    clean_id = str(request.product_id)
    
    print(f"🚀 [API] Petición recibida: Optimizar producto ID {clean_id}")
    
    try:
        # 0. Verificar que el producto no esté ya OPTIMIZED o READY_TO_PUBLISH
        status_check = await run_sync_io(
            lambda: supabase.table('shopify_products')
            .select('audit_status')
            .eq('id', clean_id)
            .single()
            .execute()
        )
        current_status = (status_check.data or {}).get('audit_status')
        if current_status in ("OPTIMIZED", "READY_TO_PUBLISH"):
            raise HTTPException(
                status_code=409,
                detail=f"Producto ya está {current_status}. No se puede re-optimizar."
            )

        # 1. Definimos el estado inicial para el flujo de LangGraph
        initial_state = {"product_id": clean_id}
        
        # 2. INVOCAMOS A LANGGRAPH (Modo Asíncrono)
        final_state = await katalog_agent.ainvoke(initial_state)
        
        # 3. Verificamos si el grafo capturó algún error en sus nodos
        if final_state.get("error"):
            print(f"⚠️ [Grafo] Error detectado en el flujo: {final_state['error']}")
            raise HTTPException(status_code=500, detail=final_state["error"])
            
        return {
            "status": "SUCCESS", 
            "message": "Asset optimized successfully", 
            "data": final_state.get("final_proposal")
        }
        
    except Exception as e:
        print(f"❌[API] Error Fatal en el proceso: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_root():
    return {
        "status": "ONLINE", 
        "brain": "Katalog AI 2.1",
        "engine": "LangGraph + Gemini 3.1 Pro + Flash Inspector"
    }
