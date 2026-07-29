import os
import re
import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices
from typing import Union
from supabase import create_client, Client

# Fix for WinError 10035 (WSAEWOULDBLOCK) on Windows with parallel async sockets.
# The default ProactorEventLoop has issues with non-blocking sockets in
# parallel async contexts. WindowsSelectorEventLoopPolicy uses select()
# instead of IOCP, which is more compatible. Linux/macOS are unaffected.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from core.graph import katalog_agent
from agents.inspector_agent import inspector_agent
from core.auth import get_current_user_id


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    print("🚀 [Servidor] Iniciando Katalog AI Brain...")
    print("🤖 [Servidor] Auto-Pilot corre aparte: python -m workers.autopilot")
    try:
        yield
    finally:
        print("✅ [Servidor] Apagado.")

# Inicializamos la API
app = FastAPI(
    title="Katalog AI Brain", 
    description="The Revenue Optimizer API - Powered by LangGraph & Gemini 3.1", 
    version="2.1",
    lifespan=lifespan
)

_allowed_origins_env = os.environ.get("ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]

if not ALLOWED_ORIGINS:
    # Fallback explícito SOLO para desarrollo local. En producción,
    # ALLOWED_ORIGINS debe estar configurada en las variables de
    # entorno del hosting (ej. "https://katalog-ai-navy.vercel.app").
    ALLOWED_ORIGINS = ["http://localhost:3000"]
    print("⚠️ [CORS] ALLOWED_ORIGINS no configurada. Usando solo localhost:3000.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
            _score = audit_data.score
            _reason = audit_data.reason
            _product_id = prod['id']
            _new_status = 'NEEDS_OPTIMIZATION' if _score < 90 else 'READY_TO_PUBLISH'

            async def _save_audit(s=_score, r=_reason, pid=_product_id, ns=_new_status):
                def _update():
                    supabase.table('shopify_products').update({
                        'seo_score_initial': s,
                        'audit_score': s,
                        'error_log': r,
                        'audit_status': ns,
                    }).eq('id', pid).execute()
                await asyncio.to_thread(_update)
            await _save_audit()
            
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
async def run_triage_scan(
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    # En lugar de hacer esperar al usuario, le pasamos el trabajo al empleado de fondo
    background_tasks.add_task(process_triage_queue)
    
    # Respondemos en 0.01 segundos
    return {
        "status": "ACCEPTED", 
        "message": "Triaje iniciado en segundo plano. Escaneando a 4 productos por minuto.",
        "user_id": user_id,
    }


# ==========================================
# 🧠 ENDPOINT 2: CIRUGÍA MAYOR (LANGGRAPH)
# ==========================================
@app.post("/api/optimize")
async def optimize_product(
    request: OptimizeRequest,
    user_id: str = Depends(get_current_user_id),
):
    clean_id = str(request.product_id)
    print(f"🚀 [API] Petición recibida: Optimizar producto ID {clean_id} (usuario {user_id})")

    try:
        # Una sola consulta trae user_id (para el chequeo de propiedad)
        # y audit_status (para el chequeo de re-optimización) que antes
        # se pedían en consultas separadas.
        status_check = await run_sync_io(
            lambda: supabase.table('shopify_products')
            .select('user_id, audit_status')
            .eq('id', clean_id)
            .single()
            .execute()
        )
        product_data = status_check.data or {}
        product_owner_id = product_data.get('user_id')
        current_status = product_data.get('audit_status')

        if not product_owner_id:
            raise HTTPException(status_code=404, detail="Producto no encontrado.")

        if str(product_owner_id) != str(user_id):
            raise HTTPException(
                status_code=403,
                detail="No tienes permiso para optimizar este producto."
            )

        if current_status in ("OPTIMIZED", "READY_TO_PUBLISH"):
            raise HTTPException(
                status_code=409,
                detail=f"Producto ya está {current_status}. No se puede re-optimizar."
            )

        initial_state = {"product_id": clean_id}
        final_state = await katalog_agent.ainvoke(initial_state)

        if final_state.get("error"):
            print(f"⚠️ [Grafo] Error detectado en el flujo: {final_state['error']}")
            raise HTTPException(status_code=500, detail=final_state["error"])

        return {
            "status": "SUCCESS",
            "message": "Asset optimized successfully",
            "data": final_state.get("final_proposal")
        }

    except HTTPException:
        # CRÍTICO: sin este except específico, el bloque de abajo
        # (except Exception) atrapa cualquier HTTPException lanzada
        # arriba -- incluyendo los 401/403/404/409 -- y la reemplaza
        # por un 500 genérico, ocultando el código de estado real al
        # cliente. Este re-raise debe ir ANTES del except Exception.
        raise
    except Exception as e:
        print(f"❌[API] Error Fatal en el proceso: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- SHOPIFY SYNC ENDPOINT ---
SHOPIFY_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]*\.myshopify\.com$")

class ShopifySyncRequest(BaseModel):
    provider: str = "shopify"

def validate_shopify_domain(value: str) -> str:
    host = value.strip().lower().removeprefix("https://").removeprefix("http://").rstrip("/")
    if not SHOPIFY_DOMAIN.fullmatch(host):
        raise HTTPException(400, "Invalid Shopify shop domain")
    return host

@app.post("/api/shopify/sync")
async def trigger_shopify_sync(
    request: ShopifySyncRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    integration = await run_sync_io(
        lambda: supabase.table("integrations_safe")
        .select("shop_url, access_token, shop_name")
        .eq("user_id", user_id)
        .eq("provider", "shopify")
        .is_("uninstalled_at", "null")
        .single()
        .execute()
    )
    row = integration.data
    if not row:
        raise HTTPException(404, "No active Shopify integration")

    shop_url = validate_shopify_domain(row["shop_url"])
    access_token = row["access_token"]
    print(f"🔄 [Sync API] Sync for user {user_id} ({shop_url})")

    async def run_sync_task():
        try:
            from core.shopify_api import sync_shopify_products
            await sync_shopify_products(
                user_id=user_id,
                shop_url=shop_url,
                access_token=access_token
            )
            print(f"✅ [Sync API] Sync completed for user {user_id}")
        except Exception as e:
            print(f"❌ [Sync API] Sync failed for user {user_id}: {e}")
            try:
                def _find_syncing_job():
                    return supabase.table("sync_jobs")\
                        .select("id")\
                        .eq("user_id", user_id)\
                        .eq("status", "syncing")\
                        .order("created_at", desc=True)\
                        .limit(1)\
                        .execute()

                res = await asyncio.to_thread(_find_syncing_job)
                if res.data:
                    job_id = res.data[0]["id"]
                    from core.shopify_api import fail_sync_job
                    await fail_sync_job(job_id, str(e))
            except Exception as inner_e:
                print(f"⚠️ [Sync API] Failed to mark job as failed: {inner_e}")

    background_tasks.add_task(run_sync_task)

    return {
        "status": "ACCEPTED",
        "message": "Shopify sync process started in the background."
    }


@app.get("/")
async def read_root():
    return {
        "status": "ONLINE", 
        "brain": "Katalog AI 2.1",
        "engine": "LangGraph + Gemini 3.1 Pro + Flash Inspector"
    }
