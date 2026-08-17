import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import END, StateGraph
from supabase import Client, create_client

from agents.critic_agent import run_critic_with_fallback
from agents.optimizer_agent import run_optimizer_with_fallback
from agents.orchestrator_agent import orchestrator_agent, OrchestratorDeps
from agents.researcher_agent import researcher_node, check_enrichment_cache, format_dossier_for_prompt
from core.helpers import (
    classify_product_type,
    calculate_precio_relativo,
    map_seo_score_to_category,
    load_skills,
    get_available_skills,
    build_product_fingerprint,
    utc_now_iso,
)
from core.quality_gate import evaluate_rewrite
from core.schemas import BrandRules, ProductContext, OrchestratorPlan
from core.shopify_tools import publish_to_shopify as publish_product_to_shopify
from core.shopify_api import get_product_taxonomy
from core.state import KatalogState

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
genai_client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

STATUS_PROCESSING = "PROCESSING"
STATUS_NEEDS_OPTIMIZATION = "NEEDS_OPTIMIZATION"
STATUS_READY_TO_PUBLISH = "READY_TO_PUBLISH"
STATUS_OPTIMIZED = "OPTIMIZED"
STATUS_ERROR = "ERROR"
STATUS_NEEDS_REVIEW = "NEEDS_REVIEW"

# ⚠️ VIVE EN DOS SISTEMAS: debe coincidir con la guarda de los triggers
# check_product_health_ins y check_product_health_upd en Supabase (ambos usan
# coalesce(NEW.consecutive_failures, 0) < 3 en su WHEN y disparan
# trigger_auto_audit()). Si el código escalara a NEEDS_REVIEW ANTES de llegar
# a este número, los triggers volverían a encolar el producto a un estado que
# el Auto-Pilot ya no toma. Si cambias este número, cambia también la
# migración 20260806000000 (fuente de verdad: la base, no el archivo).
MAX_GATE_FAILURES = 3

# ─── Cuadrantes del Candado Do-Not-Harm ──────────────────────────────────────
QUADRANT_NEEDS_OPT      = "NEEDS_OPTIMIZATION"
QUADRANT_STABLE         = "STABLE_PERFORMING"
QUADRANT_MONITORING     = "MONITORING"
QUADRANT_BENCHMARK      = "BENCHMARK"
QUADRANT_INVESTIGATE    = "INVESTIGATE_CAUSE"

_SKIP_QUADRANTS = {QUADRANT_STABLE, QUADRANT_MONITORING, QUADRANT_BENCHMARK, QUADRANT_INVESTIGATE}

BILLING_BASE_CREDITS = 1  # único costo por producto, Investigador incluido (modelo flat)

# El bucle del crítico ya consumió sus 3 iteraciones dentro de esta misma
# ejecución. Marcar el máximo (en vez de incrementar en 1) evita que el
# Auto-Pilot lo vuelva a tomar y repita el ciclo completo dos veces más.
MAX_CRITIC_ATTEMPTS = 3

TONE_PROMPT_MAP = {
    "professional": "Write in a clear, direct, results-oriented tone. Be authoritative without being stiff. Use precise language.",
    "friendly": "Write in an approachable, warm, and accessible tone. No technical jargon. Speak like a helpful friend.",
    "aspirational": "Write in an aspirational tone that sells identity and transformation. Paint a picture of the lifestyle.",
    "technical": "Write in a technical tone focused on specifications, data, and evidence. Use precise terminology.",
    "minimalist": "Write in a minimalist tone. Clean, concise, premium. Every word must earn its place. Less is more.",
    "storytelling": "Write in a storytelling tone. Tell the story behind the product — its origin, craft, and maker."
}

AUDIENCE_PROMPT_MAP = {
    "consumer": "The target audience is end consumers. They make quick, emotional purchase decisions. Lead with benefits.",
    "business": "The target audience is business buyers. They make rational, ROI-driven decisions. Lead with specs & ROI.",
    "reseller": "The target audience is resellers and distributors. They buy in volume. Lead with bulk specifications."
}

# Se inyecta al escritor cuando el Investigador no produjo dossier.
# Sin esto, el escritor copia specs de la descripción original y el Juez
# las rechaza en las 3 iteraciones: bucle garantizado, cero avance.
NO_DOSSIER_NOTICE = """
DOSSIER VERIFICADO: Sin specs verificadas.

REGLA ABSOLUTA — el Juez rechazará la propuesta si la incumples:
NO incluyas ninguna especificación técnica concreta. Esto abarca materiales,
aleaciones, composición, dimensiones, medidas, pesos, capacidades,
certificaciones, normas técnicas y cifras de rendimiento.

Esta prohibición aplica AUNQUE esos datos aparezcan en la descripción
original del producto: no han sido verificados y no puedes tratarlos como
hechos. Escribe con lenguaje cualitativo (uso previsto, beneficio, acabado,
durabilidad percibida) sin afirmar ningún dato concreto.
"""


async def _run_sync(callable_obj):
    return await asyncio.to_thread(callable_obj)


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _proposal_to_dict(proposal: Any) -> dict[str, Any]:
    if isinstance(proposal, dict):
        return proposal
    if hasattr(proposal, "model_dump"):
        return proposal.model_dump()
    return {}


def _has_proposal(proposal: Any) -> bool:
    return bool(_proposal_to_dict(proposal))


def _feedback_is_perfect(feedback: Any) -> bool:
    if feedback is None:
        return False
    if isinstance(feedback, dict):
        return feedback.get("is_perfect", False)
    return getattr(feedback, "is_perfect", False)


def _feedback_issues(feedback: Any) -> list[str]:
    if feedback is None:
        return ["El Juez no devolvió feedback legible."]
    if isinstance(feedback, dict):
        return feedback.get("issues_found", ["Errores desconocidos"])
    return getattr(feedback, "issues_found", ["Errores desconocidos"])


def _format_validation_detail_item(item: Any) -> str:
    if not isinstance(item, dict):
        return str(item)

    raw_loc = item.get("loc") or item.get("location") or item.get("field")
    if isinstance(raw_loc, (list, tuple)):
        field = ".".join(str(part) for part in raw_loc) or "__root__"
    elif raw_loc:
        field = str(raw_loc)
    else:
        field = "__root__"

    message = item.get("msg") or item.get("message") or item.get("error") or "Validation failed"
    error_type = item.get("type")
    if error_type:
        return f"field={field}: {message} (type={error_type})"
    return f"field={field}: {message}"


def _extract_validation_details(error: BaseException) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        details = getattr(current, "details", None)
        if details:
            return _serialize_validation_details(details)

        errors = getattr(current, "errors", None)
        if callable(errors):
            try:
                validation_errors = errors()
            except Exception:
                validation_errors = None
            if validation_errors:
                return _serialize_validation_details(validation_errors)

        tool_retry = getattr(current, "tool_retry", None)
        retry_content = getattr(tool_retry, "content", None)
        if retry_content:
            return _serialize_validation_details(retry_content)

        current = current.__cause__ or current.__context__

    return None


def _serialize_validation_details(details: Any) -> str:
    if isinstance(details, list):
        return "Validation details: " + "; ".join(
            _format_validation_detail_item(item) for item in details
        )

    if isinstance(details, dict):
        return "Validation details: " + json.dumps(
            details,
            ensure_ascii=False,
            default=str,
        )

    return f"Validation details: {details}"


def _format_error_for_log(error: BaseException) -> str:
    error_message = str(error)
    validation_details = _extract_validation_details(error)
    if validation_details and validation_details not in error_message:
        return f"{error_message} | {validation_details}"
    return error_message


def _error_retry_update(state: KatalogState, error_message: str) -> dict[str, Any]:
    current_retry_attempts = _to_int(state.get("retry_attempts"))
    new_retry = current_retry_attempts + 1
    if not error_message:
        error_message = ""
    err_lower = error_message.lower()

    # Caso A: Límite por Minuto (RPM)
    if ("429" in err_lower or "rate limit" in err_lower) and "day" not in err_lower:
        next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=2)).strftime('%Y-%m-%dT%H:%M:%SZ')
        audit_status = 'ERROR'
        
    # Caso B: Límite Diario Agotado (RPD)
    elif "429" in err_lower and ("day" in err_lower or "quota" in err_lower or "resourceexhausted" in err_lower):
        next_time = datetime.now(timezone.utc).replace(hour=0, minute=5, second=0, microsecond=0) + timedelta(days=1)
        next_retry_at = next_time.strftime('%Y-%m-%dT%H:%M:%SZ')
        audit_status = 'ERROR'
        
    # Caso C: Caída de Servidor Temporal
    elif any(x in err_lower for x in ["503", "502", "unavailable", "overloaded"]):
        next_retry_at = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime('%Y-%m-%dT%H:%M:%SZ')
        audit_status = 'ERROR'
        
    # Caso D: Error Fatal (No Reintentable)
    elif any(x in err_lower for x in ["401", "unauthorized", "invalid token", "validationerror", "404"]):
        new_retry = 3
        next_retry_at = None
        audit_status = 'ERROR'
        
    # Default Fallback (Otros errores no clasificados se tratan como no reintentables)
    else:
        new_retry = 3
        next_retry_at = None
        audit_status = 'ERROR'

    # Nota de Seguridad: Si el reintento alcanza o supera 3, congelamos
    if new_retry >= 3:
        next_retry_at = None

    return {
        "audit_status": audit_status,
        "error_log": error_message,
        "retry_attempts": new_retry,
        "next_retry_at": next_retry_at,
    }


def _optimization_metadata(
    state: KatalogState,
    proposal_dict: dict[str, Any],
    html: str,
) -> dict[str, Any]:
    rules = state.get("brand_rules")
    tone_from_rules = getattr(rules, "tone_voice", None) if rules else None

    return {
        "framework_used": (
            state.get("framework_used")
            or proposal_dict.get("framework_used")
            or "Katalog CRO"
        ),
        "tone_used": (
            state.get("tone_used")
            or proposal_dict.get("tone_used")
            or tone_from_rules
            or "Professional"
        ),
        "description_length": (
            state.get("description_length")
            or proposal_dict.get("description_length")
            or len(html)
        ),
    }


async def mark_product_error(
    product_id: str,
    error_message: str,
    state: KatalogState,
) -> None:
    try:
        await _run_sync(
            lambda: supabase.table("shopify_products")
            .update(_error_retry_update(state, error_message))
            .eq("id", product_id)
            .execute()
        )
    except Exception as e:
        print(f"❌ [Supabase] No se pudo registrar ERROR para producto {product_id}: {e}")


async def _refund_reservation(state: KatalogState, reason: str) -> None:
    """Reembolsa la reserva cuando no se generó valor (fallo temprano o skip do-not-harm)."""
    reservation_id = state.get("reservation_id")
    user_id = state.get("user_id")
    product_id = state.get("product_id")
    if not reservation_id or not user_id or product_id is None:
        return
    try:
        await _run_sync(
            lambda: supabase.rpc("refund_product_reservation", {
                "p_user_id": user_id,
                "p_product_id": int(product_id),
                "p_reservation_id": reservation_id,
            }).execute()
        )
        print(f"↩️ [Billing] Reserva reembolsada para producto {product_id} ({reason}).")
    except Exception as e:
        print(f"⚠️ [Billing] Error reembolsando reserva ({reason}): {e}")


async def _commit_reservation(state: KatalogState, reason: str) -> dict[str, Any] | None:
    """Confirma el cobro de la reserva (idempotente — seguro de llamar más de una vez)."""
    reservation_id = state.get("reservation_id")
    user_id = state.get("user_id")
    product_id = state.get("product_id")
    if not reservation_id or not user_id or product_id is None:
        return None
    try:
        commit_res = await _run_sync(
            lambda: supabase.rpc("commit_product_credit", {
                "p_user_id": user_id,
                "p_product_id": int(product_id),
                "p_reservation_id": reservation_id,
            }).execute()
        )
        row = (commit_res.data or [{}])[0]
        print(f"💳 [Billing] Crédito comprometido ({reason}): {row}")
        return row
    except Exception as e:
        print(f"⚠️ [Billing] Error comprometiendo crédito ({reason}): {e}")
        return None


# ==========================================
# 🚦 NODO 0: MARCAR PROCESSING
# ==========================================
async def start_processing(state: KatalogState) -> dict[str, Any]:
    product_id = str(state["product_id"])
    print(f"🚦 [Nodo 0] Producto {product_id} entra en PROCESSING...")

    try:
        current = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("audit_status, user_id, billing_state, reservation_id, credits_reserved")
            .eq("id", product_id)
            .single()
            .execute()
        )
        row = current.data or {}
        current_status = row.get("audit_status")
        user_id = row.get("user_id")

        if current_status == STATUS_OPTIMIZED:
            print(f"⛔ [Nodo 0] Producto {product_id} ya está {current_status}. Abortando para evitar amnesia de estado.")
            return {"error": f"Producto ya está {current_status}. No se puede re-procesar."}

        if not user_id:
            return {"error": f"Producto {product_id} no tiene user_id asociado."}

        billing_state = row.get("billing_state")
        existing_reservation_id = row.get("reservation_id")
        reused_committed = False

        if billing_state == "COMMITTED" and existing_reservation_id:
            # Producto ya cobrado en una ejecución anterior del grafo (retry de
            # NEEDS_OPTIMIZATION/ERROR tardío, o fast-track republicando un
            # READY_TO_PUBLISH ya pagado). NO reservar de nuevo — ya se pagó por esto.
            reservation_id = existing_reservation_id
            credits_reserved = _to_int(row.get("credits_reserved")) or BILLING_BASE_CREDITS
            reused_committed = True
            print(f"✅ [Billing] Producto {product_id} ya está COMMITTED (reservation_id={reservation_id}). Sin nueva reserva.")
        elif billing_state == "RESERVED" and existing_reservation_id:
            reservation_id = existing_reservation_id
            credits_reserved = _to_int(row.get("credits_reserved")) or BILLING_BASE_CREDITS
            print(f"♻️ [Billing] Reutilizando reserva {reservation_id} ({credits_reserved} créditos).")
        else:
            reserve_res = await _run_sync(
                lambda: supabase.rpc("reserve_or_reuse_product_credit", {
                    "p_user_id": user_id,
                    "p_product_id": int(product_id),
                    "p_base_amount": BILLING_BASE_CREDITS,
                }).execute()
            )
            reserve_row = (reserve_res.data or [{}])[0]

            if not reserve_row.get("success"):
                reason = reserve_row.get("reason", "insufficient_credits")

                if reason == "insufficient_credits":
                    await _run_sync(lambda: supabase.table("shopify_products").update({
                        "audit_status": "OUT_OF_CREDITS",
                        "error_log": "Créditos insuficientes para optimizar este producto.",
                        "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                    }).eq("id", product_id).execute())
                    print(f"💳 [Billing] Producto {product_id} → OUT_OF_CREDITS.")
                    return {"out_of_credits": True, "error": None}

                print(f"💳 [Billing] Reserva rechazada para producto {product_id}: {reason}")
                return {"error": f"No se pudo reservar crédito ({reason})."}

            reservation_id = reserve_row["reservation_id"]
            credits_reserved = reserve_row["credits_reserved"]
            print(f"💳 [Billing] Crédito(s) reservado(s): {credits_reserved} (reservation_id={reservation_id})")

        await _run_sync(lambda: supabase.table("shopify_products").update({
            "audit_status": STATUS_PROCESSING,
            "error_log": None,
            "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }, returning="representation").eq("id", product_id).execute())

        return {
            "auto_pilot_enabled": state.get("auto_pilot_enabled", False),
            "current_status": state.get("current_status") or current_status,
            "user_id": str(user_id),
            "reservation_id": str(reservation_id),
            "credits_reserved": _to_int(credits_reserved),
            "writer_invoked": reused_committed,
            "out_of_credits": False,
            "error": None,
        }
    except Exception as e:
        error_message = str(e)
        print(f"❌ [Nodo 0] Error marcando PROCESSING: {error_message}")
        return {"error": error_message}


# ==========================================
# 🛑 NODO 1: EXTRACCIÓN DE DATOS
# ==========================================
async def fetch_db_data(state: KatalogState) -> dict[str, Any]:
    product_id = str(state["product_id"])
    print(f"🔍 [Nodo 1] Buscando producto ID: {product_id}")

    if state.get("error"):
        return {}

    try:
        prod_res = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("*")
            .eq("id", product_id)
            .single()
            .execute()
        )
        product_data = prod_res.data
        if not product_data:
            raise ValueError(f"Producto {product_id} no encontrado")

        user_id = str(product_data["user_id"])
        profile_res = await _run_sync(
            lambda: supabase.table("profiles")
            .select("auto_pilot_enabled")
            .eq("id", user_id)
            .single()
            .execute()
        )
        profile_data = profile_res.data or {}
        auto_pilot_enabled = bool(profile_data.get("auto_pilot_enabled", False))
        db_status = product_data.get("audit_status")
        current_status = state.get("current_status") or db_status
        stored_proposal = product_data.get("ai_proposal")

        rules_res = await _run_sync(
            lambda: supabase.table("brand_rules")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        rules_data = rules_res.data or {}

        context = ProductContext(
            shopify_id=product_data.get("shopify_id", ""),
            current_title=product_data.get("current_title", ""),
            current_body_html=product_data.get("current_body_html", ""),
            inventory_quantity=product_data.get("inventory_quantity", 0),
            sales_last_7_days=product_data.get("sales_last_7_days", 0),
        )

        rules = BrandRules(
            tone_voice=rules_data.get("tone_voice", "professional"),
            target_audience=rules_data.get("target_audience", "consumer"),
            language=rules_data.get("language", "English"),
            forbidden_words=rules_data.get("forbidden_words", []),
            brand_dna=rules_data.get("brand_dna", ""),
            formatting_rules=rules_data.get("formatting_rules", ""),
        )

        # --- Taxonomía Predictiva de Shopify ---
        shopify_id = str(product_data.get("shopify_id") or "")
        taxonomy_text = ""
        taxonomy_ok = True
        taxonomy_attrs: list[str] = []

        if shopify_id:
            try:
                integration_res = await _run_sync(
                    lambda: supabase.table("integrations")
                    .select("shop_url, access_token")
                    .eq("user_id", user_id)
                    .eq("provider", "shopify")
                    .limit(1)
                    .execute()
                )
                integration_data = integration_res.data[0] if integration_res.data else {}
                shop_domain = integration_data.get("shop_url", "")
                encrypted_token = integration_data.get("access_token", "")

                if shop_domain and encrypted_token:
                    decrypted_res = await _run_sync(
                        lambda: supabase.rpc("decrypt_shopify_token", {"p_ciphertext_b64": encrypted_token}).execute()
                    )
                    access_token = decrypted_res.data

                    if access_token:
                        tax_res = await get_product_taxonomy(
                            shopify_numeric_id=shopify_id,
                            shop_domain=shop_domain,
                            access_token=access_token
                        )
                        taxonomy_ok = tax_res.api_ok
                        taxonomy_attrs = tax_res.attributes_with_values

                        # Estructura, no prosa interpretada: el Nodo 3 decide
                        # usando taxonomy_attributes (sin sniffing de strings).
                        if tax_res.has_verified_attributes:
                            attrs_lines = "\n".join(
                                f"- [{a}]" for a in tax_res.attributes_with_values[:50]
                            )
                            taxonomy_text = (
                                f'Categoría Shopify: "{tax_res.category_name}"'
                                f"\nAtributos requeridos:\n{attrs_lines}"
                            )
                        elif tax_res.category_name:
                            taxonomy_text = f'Categoría Shopify: "{tax_res.category_name}"'
                        else:
                            taxonomy_text = ""
            except Exception as e:
                print(f"⚠️ [Nodo 1] Error obteniendo taxonomía: {e}")
                taxonomy_ok = False

        update_data: dict[str, Any] = {
            "user_id": user_id,
            "auto_pilot_enabled": auto_pilot_enabled,
            "current_status": current_status,
            "retry_attempts": _to_int(product_data.get("retry_attempts")),
            "product_context": context,
            "brand_rules": rules,
            "taxonomy_context": taxonomy_text,
            "taxonomy_available": taxonomy_ok,
            "taxonomy_attributes": taxonomy_attrs,
        }
        if _has_proposal(stored_proposal):
            update_data["final_proposal"] = stored_proposal

        return update_data
    except Exception as e:
        error_message = str(e)
        print(f"❌ [Nodo 1] Error en DB: {error_message}")
        return {"error": error_message}


# ==========================================
# 🧠 NODO 2: RECUPERAR MEMORIA
# ==========================================
async def retrieve_memory_letta(state: KatalogState) -> dict[str, Any]:
    print("🧠 [Nodo 2] Consultando memoria a largo plazo en Letta...")

    if state.get("error"):
        return {}

    try:
        return {"letta_memory": "Insight: Focus on benefits rather than just features."}
    except Exception as e:
        error_message = str(e)
        print(f"❌ [Nodo 2] Error recuperando memoria: {error_message}")
        return {"error": error_message}


# ==========================================
# 📚 NODO 2B: CONSULTAR RAG GLOBAL
# ==========================================
async def retrieve_knowledge(state: KatalogState) -> dict[str, Any]:
    print("📚 [Nodo 2B] Consultando Knowledge Base para consejos expertos...")

    if state.get("error"):
        return {}

    context = state.get("product_context")
    if not context:
        return {"error": "No hay contexto de producto para consultar."}

    title = context.current_title
    if not title:
        print("⚠️ [Nodo 2B] Producto sin título. Saltando RAG.")
        return {"rag_knowledge": []}

    try:
        user_id = state.get("user_id")
        if not user_id:
            raise ValueError(
                "user_id ausente: no se puede consultar knowledge_base sin tenant"
            )

        result = await _run_sync(
            lambda: genai_client.models.embed_content(
                model='models/gemini-embedding-2',
                contents=title,
                config=types.EmbedContentConfig(output_dimensionality=1536)
            )
        )
        vector = result.embeddings[0].values

        rpc_res = await _run_sync(
            lambda: supabase.rpc("match_knowledge", {
                "query_embedding": vector,
                "match_threshold": 0.5,
                "match_count": 3,
                "filter": {},
                "p_user_id": user_id,
            }).execute()
        )

        matches = rpc_res.data or []
        print(f"📚 [Nodo 2B] {len(matches)} consejos recuperados de la Knowledge Base.")
        return {"rag_knowledge": matches}
    except ValueError:
        raise
    except Exception as e:
        print(f"⚠️ [Nodo 2B] Error consultando RAG: {e}")
        return {"rag_knowledge": []}


# ==========================================
# ✍️ NODO 3: LA IA ESCRIBE
# ==========================================
async def audit_and_write_pydantic(state: KatalogState) -> dict[str, Any]:
    iteration = state.get("iterations", 0)
    print(f"✍️ [Nodo 3] Gemini escribiendo (Intento {iteration + 1})...")

    if state.get("error"):
        return {}

    context = state["product_context"]
    rules = state["brand_rules"]
    memory = state.get("letta_memory", "")
    feedback = state.get("critic_feedback")
    rag = state.get("rag_knowledge", [])

    taxonomy_context = state.get("taxonomy_context", "")
    taxonomy_available = state.get("taxonomy_available", True)
    # Atributos CON valores verificados (iteración 2, emitidos por Shopify).
    # Estructura del contrato TaxonomyResult — el sniffing de strings queda
    # prohibido aquí: la rama decide por la lista, no por lo que diga el texto.
    taxonomy_attrs = state.get("taxonomy_attributes", [])

    taxonomy_injection = ""
    if taxonomy_attrs:
        # Solo se inyecta el bloque de requisitos cuando hay atributos con
        # valores verificados. Un bloque con solo nombres empujaría al
        # escritor a inventarlos (justo lo que el crítico rechaza).
        taxonomy_injection = f"""
    ## REQUISITOS TAXONÓMICOS DE SHOPIFY (MANDATORIOS PARA GOOGLE SHOPPING)
    {taxonomy_context}
    INSTRUCCIÓN: Integra cada atributo entre corchetes como prosa natural
    dentro del framework FAB/PAS. Nunca como lista técnica separada.
    Los valores vienen del catálogo de Shopify: están verificados por definición.
"""
    elif taxonomy_context:
        # Solo categoría: un hecho real del catálogo. Sirve de contexto,
        # nunca como exigencia de integrar dimensiones sin datos.
        taxonomy_injection = f"""
    ## CONTEXTO DE CATEGORÍA SHOPIFY
    {taxonomy_context}
    Esta categoría es un dato emitido por Shopify: úsala para definir QUÉ es
    el producto, no para inventar especificaciones que falten en el dossier.
"""
    elif not taxonomy_available:
        taxonomy_injection = """
    ## NOTA INTERNA: API de Shopify no disponible.
    Usa mejores prácticas generales de SEO para la descripción.
"""

    raw_tone = (rules.tone_voice or "professional").lower().strip()
    raw_audience = (rules.target_audience or "consumer").lower().strip()

    tone_instruction = TONE_PROMPT_MAP.get(raw_tone, TONE_PROMPT_MAP["professional"])
    audience_instruction = AUDIENCE_PROMPT_MAP.get(raw_audience, AUDIENCE_PROMPT_MAP["consumer"])

    prompt = f"""
    PRODUCT TO OPTIMIZE:
    - Title: {context.current_title}
    - HTML: {context.current_body_html}
    - Inventory: {context.inventory_quantity}
    - Sales (7 days): {context.sales_last_7_days}

    LONG-TERM MEMORY:
    {memory}

    BRAND RULES:
    - Tone: {tone_instruction}
    - Audience: {audience_instruction}
    - Language: {rules.language}
    - Forbidden Words: {', '.join(rules.forbidden_words)}
    - DNA: {rules.brand_dna}
    - Formats: {rules.formatting_rules}
    {taxonomy_injection}"""

    if rag:
        rag_text = "\n".join(
            f"- {r.get('content', r) if isinstance(r, dict) else r}"
            for r in rag
        )
        prompt += f"""
    CONSEJOS EXPERTOS DE VENTAS (aplicalos estrictamente):
    {rag_text}
    """

    if feedback:
        is_perf = _feedback_is_perfect(feedback)
        issues = _feedback_issues(feedback)
        if not is_perf:
            prompt += f"\n⚠️ URGENT CRITIQUE: Fix these issues immediately: {issues}"

    # ─── INYECCIÓN DEL PLAN DEL ORQUESTADOR ──────────────────────────────
    # Inyectar instrucciones del Orquestador y el dossier del Investigador
    plan          = state.get("orchestrator_plan")
    loaded_skills = state.get("loaded_skills", "")
    research_result = state.get("research_result")

    if plan:
        # Formatear dossier del Investigador (si existe)
        dossier_text = format_dossier_for_prompt(research_result) if research_result else ""
        if not dossier_text.strip():
            dossier_text = NO_DOSSIER_NOTICE

        orchestrator_section = f"""
    ════════════════════════════════════════════════════════
    INSTRUCCIONES DEL DIRECTOR DE MARKETING (MANDATORIAS)
    ════════════════════════════════════════════════════════
    DIAGNÓSTICO: {plan.diagnosis}
    PROBLEMA PRINCIPAL: {plan.primary_problem}
    ESTRATEGIA: {plan.copywriter_instructions}
"""
        orchestrator_section += f"""
    {dossier_text}
"""
        if loaded_skills:
            orchestrator_section += f"""
    ## SKILLS DE LA BIBLIOTECA (APLICAR OBLIGATORIAMENTE)
    {loaded_skills}
"""
        prompt += orchestrator_section

    try:
        result = await run_optimizer_with_fallback(prompt)
        final_data = getattr(result, "data", getattr(result, "output", result))
        return {"final_proposal": final_data, "writer_invoked": True}
    except Exception as e:
        error_message = _format_error_for_log(e)
        print(f"\u274c [Nodo 3] Error de IA: {error_message}")
        return {"error": error_message, "writer_invoked": True}


# ==========================================
# \U0001f9e0 NODO 1B: SIFÓN TOTAL (ORQUESTADOR)
# ==========================================
async def fetch_db_data_orchestrator(state: KatalogState) -> dict[str, Any]:
    """
    Nodo auxiliar: enriquece el estado con datos adicionales para el Orquestador.
    Se ejecuta DESPUÉS de fetch_db_data existente.
    Lee métricas de product_metrics y el caché de specs en paralelo.

    TIPOS DE DATOS SUPABASE (verificados via MCP):
      shopify_products.id         → BIGINT → int(product_id)
      product_metrics.product_id  → TEXT   → str(product.shopify_id)
      merchant_alerts.product_id  → BIGINT → int(product_id)
    """
    if state.get("error"):
        return {}

    # product_id es BIGINT en la DB — siempre pasamos int
    product_id = int(state["product_id"])
    user_id    = str(state.get("user_id", ""))
    product    = state.get("product_context")  # puede ser ProductContext o dict

    # Construir el dict de producto para los helpers
    if product and hasattr(product, "model_dump"):
        product_dict = product.model_dump()
        # Recuperar datos adicionales del producto completo desde la DB
        prod_res = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("*")
            .eq("id", product_id)
            .single()
            .execute()
        )
        product_dict = prod_res.data or product_dict
    else:
        product_dict = state.get("product", {})

    # Calcular fingerprint para consulta de caché
    fingerprint = build_product_fingerprint(
        product_dict.get("vendor", ""),
        product_dict.get("productType", product_dict.get("product_type", "")),
        product_dict.get("current_title", product_dict.get("title", "")),
    )

    # Consulta paralela: métricas + caché de specs
    # product_metrics.product_id es TEXT → pasamos el shopify_id como str
    shopify_id = str(product_dict.get("shopify_id", ""))

    async def _get_metrics():
        try:
            return await _run_sync(
                lambda: supabase.table("product_metrics")
                .select(
                    "orders_count_7d, orders_count_14d, orders_count_30d, "
                    "conversion_rate, performance_score, price"
                )
                .eq("product_id", shopify_id)
                .order("measured_at", desc=True)
                .limit(1)
                .execute()
            )
        except Exception as e:
            print(f"\u26a0\ufe0f [Orq] Métricas no disponibles: {e}")
            return None

    metrics_result, cached_specs = await asyncio.gather(
        _get_metrics(),
        check_enrichment_cache(fingerprint),
        return_exceptions=True,
    )

    # Construir dict de métricas seguro
    if isinstance(metrics_result, Exception) or metrics_result is None:
        metrics = {}
    elif metrics_result.data:
        metrics = metrics_result.data[0]
    else:
        metrics = {}

    if isinstance(cached_specs, Exception):
        cached_specs = None

    # Cálculos deterministas — sin LLM
    try:
        price = float(product_dict.get("price") or 0)
    except (ValueError, TypeError):
        price = 0.0

    # Precio de categoría: primero lo buscamos en métricas, sino 0
    avg_price_in_category = float(metrics.get("price", 0) or 0)

    precio_relativo    = calculate_precio_relativo(price, avg_price_in_category)
    product_type_class = classify_product_type(product_dict)
    seo_score_raw      = int(product_dict.get("seo_score_initial", 0) or 0)
    seo_score_category = map_seo_score_to_category(seo_score_raw)
    available_skills   = get_available_skills()

    print(
        f"\U0001f9e0 [Orq] Enriquecimiento: tipo={product_type_class}, "
        f"precio_rel={precio_relativo}, seo={seo_score_category}, "
        f"specs={'SI' if cached_specs else 'NO'}, skills={len(available_skills)}"
    )

    return {
        "product":            product_dict,
        "metrics":            metrics,
        "cached_specs":       cached_specs,
        "precio_relativo":    precio_relativo,
        "product_type_class": product_type_class,
        "seo_score_raw":      seo_score_raw,
        "seo_score_category": seo_score_category,
        "available_skills":   available_skills,
        "fingerprint":        fingerprint,
    }


# ==========================================
# \U0001f6ab NODO: DO-NOT-HARM (CANDADO PYTHON PURO)
# ==========================================
async def do_not_harm_check(state: KatalogState) -> dict[str, Any]:
    if state.get("error"):
        return {}

    product_id = int(state["product_id"])
    product    = state.get("product", {})

    sales_7d  = int(product.get("sales_last_7_days",  0) or 0)
    sales_30d = int(product.get("sales_last_30_days", 0) or 0)
    sales_90d = int(product.get("sales_last_90_days", 0) or 0)
    seo_score_raw = int(state.get("seo_score_raw", product.get("seo_score_initial", 0)) or 0)

    avg_weekly       = sales_90d / 13 if sales_90d > 0 else 0.0
    is_consistent    = avg_weekly >= 5
    is_viral_spike   = sales_7d > (avg_weekly * 2.5) if avg_weekly > 0 else False
    is_dead          = sales_30d < 3
    is_poor_copy     = seo_score_raw < 40
    is_good_copy     = seo_score_raw >= 70

    async def _update_status(update: dict) -> None:
        await _run_sync(
            lambda: supabase.table("shopify_products").update(update).eq("id", product_id).execute()
        )

    if is_consistent and is_poor_copy and not is_viral_spike:
        await _update_status({"audit_status": QUADRANT_STABLE})
        await _refund_reservation(state, "do_not_harm:STABLE_PERFORMING")
        print(f"🚫 [Do-Not-Harm] {product_id} → STABLE_PERFORMING (avg_weekly={avg_weekly:.1f})")
        return {"product_quadrant": QUADRANT_STABLE}

    if is_viral_spike:
        monitoring_since_str = product.get("monitoring_since")

        if monitoring_since_str:
            try:
                monitoring_since = datetime.fromisoformat(monitoring_since_str.replace("Z", "+00:00"))
                weeks_elapsed = (datetime.now(timezone.utc) - monitoring_since).days / 7
            except (ValueError, TypeError):
                weeks_elapsed = 0.0

            if weeks_elapsed >= 2:
                await _update_status({"monitoring_since": None})
                print(f"⏰ [Do-Not-Harm] {product_id}: pico terminó, optimizando")
            else:
                await _refund_reservation(state, "do_not_harm:MONITORING")
                print(f"📊 [Do-Not-Harm] {product_id} → MONITORING ({weeks_elapsed:.1f} semanas)")
                return {"product_quadrant": QUADRANT_MONITORING}
        else:
            now_iso = datetime.now(timezone.utc).isoformat()
            await _update_status({"audit_status": QUADRANT_MONITORING, "monitoring_since": now_iso})
            await _refund_reservation(state, "do_not_harm:MONITORING_START")
            print(f"📊 [Do-Not-Harm] {product_id} → MONITORING (inicio)")
            return {"product_quadrant": QUADRANT_MONITORING}

    if is_consistent and is_good_copy:
        await _update_status({"audit_status": QUADRANT_BENCHMARK})
        await _refund_reservation(state, "do_not_harm:BENCHMARK")
        print(f"🏆 [Do-Not-Harm] {product_id} → BENCHMARK")
        return {"product_quadrant": QUADRANT_BENCHMARK}

    if is_dead and is_good_copy:
        await _update_status({"audit_status": QUADRANT_INVESTIGATE})
        await _refund_reservation(state, "do_not_harm:INVESTIGATE_CAUSE")
        print(f"🔍 [Do-Not-Harm] {product_id} → INVESTIGATE_CAUSE")
        return {"product_quadrant": QUADRANT_INVESTIGATE}

    print(f"🎯 [Do-Not-Harm] {product_id} → NEEDS_OPTIMIZATION")
    return {"product_quadrant": QUADRANT_NEEDS_OPT}


# ==========================================
# \U0001f4cb NODO: ORQUESTADOR (PLAN DE VUELO)
# ==========================================
async def orchestrator_node(state: KatalogState) -> dict[str, Any]:
    """
    Nodo del Orquestador: diseña el Plan de Vuelo.
    Si falla, usa un plan conservador hardcodeado.
    Nunca rompe el grafo.

    TIPOS DE DATOS SUPABASE:
      shopify_products.id         → BIGINT → int(product_id)
      merchant_alerts.product_id  → BIGINT → int(product_id)
    """
    if state.get("error"):
        return {}

    product_id = int(state["product_id"])
    user_id    = str(state.get("user_id", ""))

    deps = OrchestratorDeps(
        product            = state.get("product", {}),
        metrics            = state.get("metrics", {}),
        cached_specs       = state.get("cached_specs"),
        precio_relativo    = state.get("precio_relativo", 1.0),
        product_type_class = state.get("product_type_class", "GENERIC"),
        available_skills   = state.get("available_skills", []),
        seo_score_category = state.get("seo_score_category", "POOR"),
        seo_score_raw      = state.get("seo_score_raw", 0),
    )

    try:
        result = await orchestrator_agent.run(
            "Diagnostica este producto y genera el Plan de Vuelo.",
            deps=deps,
        )
        plan = result.output   # NO result.data — API correcta de PydanticAI v2
        print(f"\U0001f4cb [Orquestador] Plan generado: {plan.primary_problem} / {plan.product_quadrant}")

    except Exception as e:
        print(f"\u274c [Orquestador] Falló para producto {product_id}: {e}")
        # Plan conservador — el sistema nunca se detiene
        plan = OrchestratorPlan(
            diagnosis                 = "Orquestador no disponible — plan conservador",
            primary_problem           = "poor_copy",
            activate_researcher_agent = False,
            research_instructions     = None,
            copywriter_instructions   = (
                "Mejora el copy usando buenas prácticas de SEO y e-commerce. "
                "Estructura: H2 con keyword + párrafo de beneficios + 3 bullets. "
                "NO inventes especificaciones técnicas."
            ),
            judge_instructions        = (
                "Verifica que el copy no contenga afirmaciones técnicas sin respaldo. "
                "Rechaza si hay garantías o materiales inventados."
            ),
            skills_to_inject          = [],
            do_not_harm_triggered     = False,
            product_quadrant          = QUADRANT_NEEDS_OPT,
            fact_anchored_alert       = False,
            merchant_alert_message    = None,
        )

    # Guardar diagnóstico en Supabase para auditoría
    # shopify_products.id es BIGINT → int(product_id)
    await _run_sync(
        lambda: supabase.table("shopify_products")
        .update({
            "orchestrator_diagnosis": plan.model_dump(),
            "audit_status":           plan.product_quadrant,
        })
        .eq("id", product_id)
        .execute()
    )

    # Si hay alerta para el comerciante — guardar en merchant_alerts
    # merchant_alerts.product_id es BIGINT → int(product_id)
    if plan.fact_anchored_alert and plan.merchant_alert_message and user_id:
        try:
            await _run_sync(
                lambda: supabase.table("merchant_alerts").insert({
                    "user_id":    user_id,
                    "product_id": product_id,   # BIGINT → int
                    "alert_type": "missing_value_justification",
                    "message":    plan.merchant_alert_message,
                    "status":     "unread",
                }).execute()
            )
            print(f"\U0001f514 [Orquestador] Alerta de comerciante guardada para producto {product_id}")
        except Exception as e:
            print(f"\u26a0\ufe0f [Orquestador] No se pudo guardar la alerta: {e}")

    # Pre-cargar skills para el Redactor
    loaded_skills = load_skills(plan.skills_to_inject)

    return {
        "orchestrator_plan": plan,
        "loaded_skills":     loaded_skills,
    }


# ==========================================
# ⚖️ NODO 4: EL JUEZ
# ==========================================
async def review_proposal(state: KatalogState) -> dict[str, Any]:
    iteration = state.get("iterations", 0) + 1
    print(f"⚖️ [Nodo 4] Juez auditando propuesta (Intento {iteration})...")

    proposal = state.get("final_proposal")
    if state.get("error"):
        return {"iterations": iteration}
    if not proposal:
        return {"error": "No hay propuesta para evaluar.", "iterations": iteration}

    rules = state["brand_rules"]
    raw_tone = (rules.tone_voice or "professional").lower().strip()
    tone_instruction = TONE_PROMPT_MAP.get(raw_tone, TONE_PROMPT_MAP["professional"])

    prompt = f"""
    BRAND RULES TO ENFORCE:
    - Tone: {tone_instruction}
    - Forbidden Words: {', '.join(rules.forbidden_words)}
    - Brand DNA: {rules.brand_dna}

    AI PROPOSAL TO REVIEW:
    {proposal}

    Verify if the proposal follows all rules strictly.
    If it uses ANY forbidden words, reject it (is_perfect=False) and list them.
    If the title is over 70 chars, reject it.
    Provide actionable feedback for the writer to fix it.
    """

    # Inyectar instrucciones del Juez desde el Plan del Orquestador
    plan = state.get("orchestrator_plan")
    if plan and plan.judge_instructions:
        prompt += f"""

    INSTRUCCIONES ESPECÍFICAS DEL DIRECTOR (VERIFICAR OBLIGATORIAMENTE):
    {plan.judge_instructions}

    DOSSIER VERIFICADO (fuente única de hechos — cualquier afirmación técnica
    que no esté en este dossier DEBE ser rechazada):
    {state.get('cached_specs') or 'Sin specs verificadas'}
    """

    try:
        result = await run_critic_with_fallback(prompt, proposal=proposal, rules=rules)
        feed_data = getattr(result, "data", getattr(result, "output", result))
        print(f"🔎 [Nodo 4] Veredicto del Juez: {feed_data}")
        return {"critic_feedback": feed_data, "iterations": iteration}
    except Exception as e:
        error_message = _format_error_for_log(e)
        print(f"❌ [Nodo 4] Error del Juez: {error_message}")
        return {"error": error_message, "iterations": iteration}


# ==========================================
# 🔀 ENRUTADOR DE CALIDAD
# ==========================================
def should_continue(state: KatalogState) -> str:
    if state.get("error"):
        return "error_handler"

    feedback = state.get("critic_feedback")
    iterations = state.get("iterations", 0)

    print(f"🚦 [Enrutador] Analizando veredicto del Juez (Intento {iterations})...")

    if feedback is None:
        return "error_handler"

    if _feedback_is_perfect(feedback):
        print("🟢 [Decisión] Calidad aprobada. Pasando a READY_TO_PUBLISH.")
        return "save_db"

    issues = _feedback_issues(feedback)
    if iterations >= 3:
        print(f"🟠 [Decisión] Límite alcanzado. Marcando NEEDS_OPTIMIZATION: {issues}")
        return "needs_optimization"

    print(f"🔴 [Decisión] Errores detectados: {issues}. Devolviendo al Escritor...")
    return "ai_writer"


# ==========================================
# 💾 NODO 5: GUARDAR READY_TO_PUBLISH
# ==========================================
async def save_to_supabase(state: KatalogState) -> dict[str, Any]:
    print("💾 [Nodo 5] Guardando propuesta y ejecutando quality gate...")
    proposal = state.get("final_proposal")

    if state.get("error"):
        return {}
    if not proposal:
        return {"error": "No hay propuesta aprobada para guardar."}

    try:
        proposal_dict = _proposal_to_dict(proposal)
        proposal_dict.pop("audit_score", None)  # el escritor ya no se autocalifica
        audit_log_data = list(proposal_dict.get("audit_log", []))

        # Obtener texto original del estado (cargado en Nodo 1 vía product_context)
        context = state.get("product_context")
        if context and isinstance(context, ProductContext):
            old_title = context.current_title
            old_body_html = context.current_body_html
        elif context and isinstance(context, dict):
            old_title = context.get("current_title", "")
            old_body_html = context.get("current_body_html", "")
        else:
            old_title, old_body_html = "", ""

        # El inspector necesita una REFERENCIA de identidad para juzgar
        # relevancia. Caso real: el producto 1004 (snowboard) recibio una
        # propuesta titulada "Estabilizador de Video Profesional para
        # Camara y Cine" y el gate anterior no pudo detectarlo porque
        # nunca supo de que producto se trataba.
        verdict = await evaluate_rewrite(
            current_title=old_title,
            current_body_html=old_body_html,
            candidate_title=proposal_dict.get("new_title"),
            candidate_body_html=proposal_dict.get("new_body_html"),
            vendor=(state.get("product") or {}).get("vendor"),
            tags=(state.get("product") or {}).get("tags"),
            forbidden_words=state["brand_rules"].forbidden_words if state.get("brand_rules") else None,
        )
        audit_log_data.append(verdict.as_log_line())
        print(f"⚖️ [Gate] {verdict.as_log_line()}")

        next_status = STATUS_READY_TO_PUBLISH if verdict.passed else STATUS_NEEDS_OPTIMIZATION

        # Leer consecutive_failures actual de la fila (necesario para incrementar)
        cf_res = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("consecutive_failures")
            .eq("id", state["product_id"])
            .single()
            .execute()
        )
        current_cf = (cf_res.data or {}).get("consecutive_failures") or 0

        if verdict.passed:
            next_cf = 0
            next_status = STATUS_READY_TO_PUBLISH
        else:
            next_cf = current_cf + 1
            # Escalar al tercer rechazo seguido: el circuito deja de re-encolar
            # y pasa a revisión humana. Sin esto el producto quedaría encolado
            # para siempre (el trigger re-encola mientras cf < MAX_GATE_FAILURES).
            next_status = (
                STATUS_NEEDS_REVIEW
                if next_cf >= MAX_GATE_FAILURES
                else STATUS_NEEDS_OPTIMIZATION
            )

        update_payload: dict[str, Any] = {
            "ai_proposal": proposal_dict,
            "audit_score": verdict.audit_score,
            "audit_log": audit_log_data,
            "audit_status": next_status,
            "last_audit_at": utc_now_iso(),
            "error_log": None,
            "retry_attempts": 0,
            "next_retry_at": None,
            "consecutive_failures": next_cf,
        }

        if not verdict.passed:
            update_payload["last_failure_at"] = utc_now_iso()

        await _run_sync(
            lambda: supabase.table("shopify_products")
            .update(update_payload)
            .eq("id", state["product_id"])
            .execute()
        )

        if verdict.passed:
            # 💰 Único punto de cobro: el LLM ya trabajó y el gate aprobó.
            # Publicar después (ahora o en una semana) es gratis.
            await _commit_reservation(state, "optimization_complete")
            print("   [Nodo 5] Propuesta guardada, gate aprobado, crédito comprometido.")
            return {"status": STATUS_READY_TO_PUBLISH, "retry_attempts": 0}
        else:
            # Gate rechazó: no hay entregable, no se cobra.
            # El costo de tokens lo absorbe el negocio: es un fallo de
            # calidad nuestro, no un servicio prestado al cliente.
            await _refund_reservation(state, "gate_rejected")
            print(f"  [Nodo 5] Gate rechazó: {verdict.as_log_line()} Crédito devuelto, marcado {next_status}.")
            return {"status": next_status, "retry_attempts": 0}

    except Exception as e:
        error_message = _format_error_for_log(e)
        print(f"❌ [Nodo 5] Error en gate/guardado: {error_message}")
        return {"error": error_message}


# ==========================================
# 🟠 NODO 5B: NECESITA OPTIMIZACIÓN
# ==========================================
async def mark_needs_optimization(state: KatalogState) -> dict[str, Any]:
    product_id = str(state["product_id"])
    feedback = state.get("critic_feedback")
    proposal = state.get("final_proposal")
    issues = _feedback_issues(feedback)
    error_log = "; ".join(issues)

    try:
        # Contador primero: el fallo del bucle del crítico es también un fallo
        # de producción de la IA. Si no contara, un producto que siempre falla
        # por este camino quedaría en loop infinito sin llegar a revisión humana.
        cf_res = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("consecutive_failures")
            .eq("id", product_id)
            .single()
            .execute()
        )
        current_cf = (cf_res.data or {}).get("consecutive_failures") or 0
        next_cf = current_cf + 1
        next_status = (
            STATUS_NEEDS_REVIEW
            if next_cf >= MAX_GATE_FAILURES
            else STATUS_NEEDS_OPTIMIZATION
        )

        update_data: dict[str, Any] = {
            "audit_status": next_status,
            "error_log": error_log,
            # 0, no MAX_CRITIC_ATTEMPTS: el filtro de elegibilidad del Auto-Pilot
            # exige retry_attempts < 3, asi que escribir 3 aqui saca al producto de
            # la cola con un solo fallo de calidad y le impide escalar a
            # NEEDS_REVIEW. La escalada la gobierna consecutive_failures (ver
            # MAX_GATE_FAILURES); retry_attempts es backoff de errores transitorios
            # del proveedor, no un bloqueo de calidad. El nodo 5 ya escribe 0.
            "retry_attempts": 0,
            "next_retry_at": None,
            "consecutive_failures": next_cf,
            "last_failure_at": utc_now_iso(),
        }

        if proposal:
            proposal_dict = _proposal_to_dict(proposal)
            proposal_dict.pop("audit_score", None)  # el escritor ya no se autocalifica
            update_data["ai_proposal"] = proposal_dict
            # audit_score NO se toca aquí: este nodo es el camino de error del Critic
            # (iteraciones excedidas), no tiene gate independiente. El score se mantendrá
            # como el que dejó el gate si ya corrió, o 0 por default de la columna.
            update_data["audit_log"] = proposal_dict.get("audit_log", [])

        await _run_sync(
            lambda: supabase.table("shopify_products")
            .update(update_data)
            .eq("id", product_id)
            .execute()
        )

        # El crítico no aprobó: no hay entregable, no se cobra.
        # El costo de tokens lo absorbe el negocio, no el cliente.
        await _refund_reservation(state, "needs_optimization")

        print(f"🟠 [Nodo 5B] Producto {product_id} marcado como {next_status} (crédito devuelto).")
        return {"status": next_status}
    except Exception as e:
        error_message = str(e)
        print(f"❌ [Nodo 5B] Error marcando NEEDS_OPTIMIZATION: {error_message}")
        return {"error": error_message}


# ==========================================
# 🚀 NODO 6: PUBLICAR EN SHOPIFY
# ==========================================
async def publish_to_shopify_node(state: KatalogState) -> dict[str, Any]:
    print("🚀 [Nodo 6] Auto-Pilot publicando en Shopify...")

    product_id = str(state["product_id"])
    proposal = state.get("final_proposal")
    context = state.get("product_context")
    user_id = state.get("user_id")

    if state.get("error"):
        return {}
    if not proposal or not context or not user_id:
        return {"error": "Faltan datos para publicar en Shopify."}

    proposal_dict = _proposal_to_dict(proposal)
    title = proposal_dict.get("new_title", "")
    html = proposal_dict.get("new_body_html", "")
    metadata = _optimization_metadata(state, proposal_dict, html)

    if not title or not html:
        return {"error": "La propuesta aprobada no contiene título o descripción HTML."}

    try:
        integration_res = await _run_sync(
            lambda: supabase.table("integrations")
            .select("shop_url,access_token")
            .eq("user_id", user_id)
            .eq("provider", "shopify")
            .limit(1)
            .execute()
        )
        integration_data = integration_res.data[0] if integration_res.data else {}
        if not integration_data:
            raise ValueError(f"No hay integración Shopify para usuario {user_id}")

        encrypted_token = integration_data.get("access_token", "")
        if not encrypted_token:
            raise ValueError("No se encontró access_token en integración Shopify")

        decrypted_res = await _run_sync(
            lambda: supabase.rpc("decrypt_shopify_token", {"p_ciphertext_b64": encrypted_token}).execute()
        )
        access_token = decrypted_res.data

        await publish_product_to_shopify(
            shop_url=integration_data.get("shop_url", ""),
            access_token=access_token,
            product_shopify_id=context.shopify_id,
            title=title,
            html=html,
        )

        published_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        await _run_sync(lambda: supabase.table("shopify_products").update({
            "audit_status": STATUS_OPTIMIZED,
            "current_title": title,
            "current_body_html": html,
            "last_audit_at": published_at,
            "error_log": None,
            "retry_attempts": 0,
            "next_retry_at": None,
        }, returning="representation").eq("id", product_id).execute())

        await _run_sync(lambda: supabase.table("optimizations").insert({
            "user_id": user_id,
            "product_id": product_id,
            "title_generated": title,
            "description_generated": html,
            "title_previous": context.current_title,
            "description_previous": context.current_body_html,
            "framework_used": metadata["framework_used"],
            "tone_used": metadata["tone_used"],
            "description_length": metadata["description_length"],
            "status": "published",
        }).execute())

        # ✅ Sin lógica de créditos aquí. El cobro ya ocurrió en save_to_supabase.
        print(f"✅ [Nodo 6] Producto {product_id} publicado y marcado como OPTIMIZED.")
        return {"status": STATUS_OPTIMIZED, "retry_attempts": 0}
    except Exception as e:
        error_message = _format_error_for_log(e)
        print(f"❌ [Nodo 6] Error publicando producto {product_id}: {error_message}")
        return {"error": error_message}


# ==========================================
# 🧯 NODO GLOBAL DE ERROR
# ==========================================
async def error_handler(state: KatalogState) -> dict[str, Any]:
    product_id = state["product_id"]
    fallback_message = (
        "El Juez no devolvió feedback legible."
        if state.get("critic_feedback") is None
        else "Error desconocido en LangGraph."
    )
    error_message = str(state.get("error") or fallback_message)
    print(f"🧯 [Error Handler] Marcando producto {product_id} como ERROR: {error_message}")

    user_id = state.get("user_id")
    reservation_id = state.get("reservation_id")
    writer_invoked = bool(state.get("writer_invoked"))

    if reservation_id and user_id:
        try:
            # Falle temprano o tarde, si no se publicó no se cobra.
            # Un crash del pipeline es un fallo nuestro, no un servicio prestado.
            await _refund_reservation(state, "pipeline_failure")
        except Exception as billing_error:
            print(f"⚠️ [Billing] Error ajustando créditos tras fallo: {billing_error}")

        try:
            fail_res = await _run_sync(
                lambda: supabase.rpc("record_product_failure_and_maybe_compensate", {
                    "p_user_id": user_id,
                    "p_product_id": int(product_id),
                }).execute()
            )
            fail_row = (fail_res.data or [{}])[0]
            if fail_row.get("compensation_granted"):
                print(f"🎁 [Billing] Compensación automática otorgada a usuario {user_id}.")
        except Exception as e:
            print(f"⚠️ [Billing] Error registrando fallo consecutivo: {e}")

    await mark_product_error(product_id, error_message, state)
    return {"status": STATUS_ERROR}



# ─── ENRUTADORES ORIGINALES (flujo base) ─────────────────────────────────────

def route_after_start(state: KatalogState) -> str:
    if state.get("out_of_credits"):
        return "end"
    return "error_handler" if state.get("error") else "fetch_data"


def route_after_fetch(state: KatalogState) -> str:
    if state.get("error"):
        return "error_handler"
    if (
        state.get("current_status") == STATUS_READY_TO_PUBLISH
        and state.get("auto_pilot_enabled") is True
        and _has_proposal(state.get("final_proposal"))
    ):
        print("⚡ [Fast-Track] Producto READY_TO_PUBLISH con propuesta existente. Saltando IA y publicando.")
        return "publish_to_shopify"
    return "memory"


def route_after_memory(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "retrieve_knowledge"


def route_after_knowledge(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "ai_writer"


def route_after_save(state: KatalogState) -> str:
    """Decide si el producto merece publicarse tras el quality gate (Nodo 5).

    El gate puntúa el texto viejo y el nuevo con el mismo juez en la misma
    corrida. Si el nuevo no lo supera por el margen mínimo, save_to_supabase
    escribe NEEDS_OPTIMIZATION y devuelve el crédito, y su retorno se mergea
    en el estado. Lista blanca: solo READY_TO_PUBLISH publica; cualquier otro
    status bloquea (fallo cerrado: el default es no escribir en la tienda).

    Sin esta guarda, la arista condicional anterior publicaba con solo
    auto_pilot_enabled, ignorando el veredicto del gate. Verificado en
    producción el 2026-08-08 con los productos 1009 (delta -20) y 998.
    """
    if state.get("error"):
        return "error_handler"

    status = state.get("status")

    if (
        state.get("auto_pilot_enabled", False)
        and status == STATUS_READY_TO_PUBLISH
    ):
        return "publish_to_shopify"

    print(f"🛑 [Enrutador] Publicación bloqueada (status={status}). No se publica.")
    return "end"


def route_after_publish(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "end"


def route_after_needs_optimization(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "end"


# ─── ENRUTADORES DEL ORQUESTADOR ───────────────────────────────────────────────────────

def route_after_enrich(state: KatalogState) -> str:
    """Después del enriquecimiento — ir al candado o al manejador de errores."""
    if state.get("error"):
        return "error_handler"
    return "do_not_harm"


def route_after_do_not_harm(state: KatalogState) -> str:
    """
    Si el producto no necesita optimización → END inmediato.
    El Orquestador nunca ve estos productos.
    """
    if state.get("error"):
        return "error_handler"
    quadrant = state.get("product_quadrant", QUADRANT_NEEDS_OPT)
    if quadrant in _SKIP_QUADRANTS:
        return "end"
    return "orchestrator"


def route_after_orchestrator(state: KatalogState) -> str:
    """
    El Orquestador ya decidió — Python ejecuta su decisión.
    """
    if state.get("error"):
        return "error_handler"
    plan = state.get("orchestrator_plan")
    if not plan:
        return "ai_writer"   # fallback seguro
    if plan.activate_researcher_agent:
        return "researcher"
    return "ai_writer"


# ─── GRAFO COMPLETO ──────────────────────────────────────────────────────────────────

def build_graph():
    workflow = StateGraph(KatalogState)

    # ─ Nodos existentes ────────────────────────────────────────────────────────────
    workflow.add_node("start_processing", start_processing)
    workflow.add_node("fetch_data",       fetch_db_data)
    workflow.add_node("memory",           retrieve_memory_letta)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("ai_writer",        audit_and_write_pydantic)
    workflow.add_node("critic",           review_proposal)
    workflow.add_node("save_db",          save_to_supabase)
    workflow.add_node("needs_optimization", mark_needs_optimization)
    workflow.add_node("publish_to_shopify", publish_to_shopify_node)
    workflow.add_node("error_handler",    error_handler)

    # ─ Nodos nuevos: Orquestador Layer ───────────────────────────────────────────
    workflow.add_node("enrich_for_orchestrator", fetch_db_data_orchestrator)
    workflow.add_node("do_not_harm",     do_not_harm_check)
    workflow.add_node("orchestrator",    orchestrator_node)
    workflow.add_node("researcher",      researcher_node)

    # ─ Entry point ───────────────────────────────────────────────────────────────
    workflow.set_entry_point("start_processing")

    # ─ Flujo base (pre-orquestador) ────────────────────────────────────────────
    workflow.add_conditional_edges(
        "start_processing",
        route_after_start,
        {"fetch_data": "fetch_data", "error_handler": "error_handler", "end": END},
    )
    workflow.add_conditional_edges(
        "fetch_data",
        route_after_fetch,
        {
            "memory":           "memory",
            "publish_to_shopify": "publish_to_shopify",
            "error_handler":    "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "memory",
        route_after_memory,
        {"retrieve_knowledge": "retrieve_knowledge", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "retrieve_knowledge",
        route_after_knowledge,
        # Ahora → enriquecer antes del writer (capa del orquestador)
        {"ai_writer": "enrich_for_orchestrator", "error_handler": "error_handler"},
    )

    # ─ Capa del Orquestador ─────────────────────────────────────────────────────
    workflow.add_conditional_edges(
        "enrich_for_orchestrator",
        route_after_enrich,
        {"do_not_harm": "do_not_harm", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "do_not_harm",
        route_after_do_not_harm,
        {
            "orchestrator":  "orchestrator",
            "end":           END,
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "researcher":    "researcher",
            "ai_writer":     "ai_writer",
            "error_handler": "error_handler",
        },
    )
    # El Investigador siempre va al Redactor después
    workflow.add_edge("researcher", "ai_writer")

    # ─ Bucle de calidad (existente) ──────────────────────────────────────────────
    workflow.add_edge("ai_writer", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "save_db":           "save_db",
            "ai_writer":         "ai_writer",
            "needs_optimization": "needs_optimization",
            "error_handler":     "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "save_db",
        route_after_save,
        {
            "publish_to_shopify": "publish_to_shopify",
            "error_handler":      "error_handler",
            "end":                END,
        },
    )
    workflow.add_conditional_edges(
        "publish_to_shopify",
        route_after_publish,
        {"error_handler": "error_handler", "end": END},
    )
    workflow.add_conditional_edges(
        "needs_optimization",
        route_after_needs_optimization,
        {"error_handler": "error_handler", "end": END},
    )
    workflow.add_edge("error_handler", END)

    return workflow.compile()


katalog_agent = build_graph()

