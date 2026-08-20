import os
import re
import uuid
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
from core.auth import get_current_user_id
from core.helpers import utc_now_iso


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

# Inicializamos Supabase Globalmente para el sync y la cirugía mayor
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
# 🧠 ENDPOINT 1: CIRUGÍA MAYOR (LANGGRAPH) — 202 Accepted + Polling
# ==========================================
async def run_grafo_background(job_id: str, product_id: str) -> None:
    """Ejecuta el grafo en segundo plano, DESPUÉS de que la respuesta 202 ya
    viaja al cliente. El request HTTP nunca espera al grafo: el estado se
    persiste en shopify_products.audit_status y el cliente lo consulta por
    polling.

    - Éxito: el propio grafo escribe el estado final (READY_TO_PUBLISH,
      NEEDS_OPTIMIZATION, cuadrantes, etc.) en save_db.
    - Falla: escribimos ERROR aquí, con guard atómico sobre PROCESSING para
      no pisar un estado que el grafo ya alcanzó (ej. OUT_OF_CREDITS).
    No hay scoped_session que cerrar: el cliente Supabase es global y todos
    los llamados van por asyncio.to_thread.
    """
    try:
        print(f"🎬 [Job {job_id}] Grafo iniciado para producto {product_id}")
        final_state = await katalog_agent.ainvoke({"product_id": product_id})

        if final_state.get("error"):
            raise RuntimeError(str(final_state["error"]))

        print(f"✅ [Job {job_id}] Grafo terminado para producto {product_id}")
    except Exception as e:
        error_message = str(e)
        print(f"❌ [Job {job_id}] Grafo falló para producto {product_id}: {error_message}")
        try:
            await run_sync_io(
                lambda: supabase.table('shopify_products')
                .update({
                    "audit_status": "ERROR",
                    "error_log": error_message,
                    "processing_heartbeat_at": None,
                })
                .eq("id", product_id)
                .eq("audit_status", "PROCESSING")
                .execute()
            )
            print(f"❌ [Job {job_id}] Estado ERROR persistido para producto {product_id}")
        except Exception as write_error:
            print(f"⚠️ [Job {job_id}] No se pudo persistir ERROR para {product_id}: {write_error}")


@app.post("/api/optimize")
async def optimize_product(
    request: OptimizeRequest,
    background_tasks: BackgroundTasks,
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

        if current_status in ("OPTIMIZED", "READY_TO_PUBLISH", "PROCESSING"):
            raise HTTPException(
                status_code=409,
                detail=f"Producto ya está {current_status}. No se puede optimizar ahora."
            )

        if not current_status:
            raise HTTPException(
                status_code=409,
                detail="El producto no tiene un estado de auditoría válido."
            )

        # Claim atómico (compare-and-swap): solo el request que gane la
        # condición pasa el producto a PROCESSING. Si dos peticiones llegan
        # simultáneas, la segunda actualiza 0 filas → 409 Conflict.
        job_id = str(uuid.uuid4())
        claimed = await run_sync_io(
            lambda: supabase.table('shopify_products')
            .update({
                "audit_status": "PROCESSING",
                "processing_heartbeat_at": utc_now_iso(),
            }, returning="representation")
            .eq("id", clean_id)
            .eq("audit_status", current_status)
            .execute()
        )
        if not (claimed.data or []):
            raise HTTPException(
                status_code=409,
                detail="El producto ya está en cola de optimización. Espera a que termine."
            )

        # El grafo corre en segundo plano; el 202 sale ya, en <1s.
        background_tasks.add_task(run_grafo_background, job_id, clean_id)

        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Optimization queued",
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
        lambda: supabase.table("integrations")
        .select("shop_url, access_token, shop_name")
        .eq("user_id", user_id)
        .eq("provider", "shopify")
        .is_("uninstalled_at", None)
        .limit(1)
        .execute()
    )
    row = integration.data[0] if integration.data else None
    if not row:
        raise HTTPException(404, "No active Shopify integration")

    shop_url = validate_shopify_domain(row["shop_url"])
    encrypted_token = row.get("access_token", "")
    if not encrypted_token:
        raise HTTPException(404, "No access token found for Shopify integration")

    decrypted_res = await run_sync_io(
        lambda: supabase.rpc("decrypt_shopify_token", {"p_ciphertext_b64": encrypted_token}).execute()
    )
    access_token = decrypted_res.data
    if not access_token:
        raise HTTPException(500, "Failed to decrypt Shopify access token")

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
