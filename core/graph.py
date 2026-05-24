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
from core.schemas import BrandRules, ProductContext
from core.shopify_tools import publish_to_shopify as publish_product_to_shopify
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


async def charge_profile_credit(user_id: str) -> bool:
    try:
        await _run_sync(
            lambda: supabase.rpc(
                "increment_profile_credits_used",
                {"p_user_id": user_id},
            ).execute()
        )
        print(f"💳 [Créditos] Crédito consumido para usuario {user_id}.")
        return True
    except Exception as e:
        print(f"❌ [Créditos] Error al cobrar crédito al usuario {user_id}: {e}")
        return False


# ==========================================
# 🚦 NODO 0: MARCAR PROCESSING
# ==========================================
async def start_processing(state: KatalogState) -> dict[str, Any]:
    product_id = str(state["product_id"])
    print(f"🚦 [Nodo 0] Producto {product_id} entra en PROCESSING...")

    try:
        current = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("audit_status")
            .eq("id", product_id)
            .single()
            .execute()
        )
        current_status = (current.data or {}).get("audit_status")
        if current_status in (STATUS_OPTIMIZED, STATUS_READY_TO_PUBLISH):
            print(f"⛔ [Nodo 0] Producto {product_id} ya está {current_status}. Abortando para evitar amnesia de estado.")
            return {"error": f"Producto ya está {current_status}. No se puede re-procesar."}

        await _run_sync(lambda: supabase.table("shopify_products").update({
            "audit_status": STATUS_PROCESSING,
            "error_log": None,
            "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }, returning="representation").eq("id", product_id).execute())

        return {
            "auto_pilot_enabled": state.get("auto_pilot_enabled", False),
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
            tone_voice=rules_data.get("tone_voice", "Professional"),
            target_audience=rules_data.get("target_audience", "General"),
            language=rules_data.get("language", "English"),
            forbidden_words=rules_data.get("forbidden_words", []),
            brand_dna=rules_data.get("brand_dna", ""),
            formatting_rules=rules_data.get("formatting_rules", ""),
        )

        return {
            "user_id": user_id,
            "auto_pilot_enabled": auto_pilot_enabled,
            "retry_attempts": _to_int(product_data.get("retry_attempts")),
            "product_context": context,
            "brand_rules": rules,
        }
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
            }).execute()
        )

        matches = rpc_res.data or []
        print(f"📚 [Nodo 2B] {len(matches)} consejos recuperados de la Knowledge Base.")
        return {"rag_knowledge": matches}
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

    prompt = f"""
    PRODUCT TO OPTIMIZE:
    - Title: {context.current_title}
    - HTML: {context.current_body_html}
    - Inventory: {context.inventory_quantity}
    - Sales (7 days): {context.sales_last_7_days}

    LONG-TERM MEMORY:
    {memory}

    BRAND RULES:
    - Tone: {rules.tone_voice}
    - Audience: {rules.target_audience}
    - Language: {rules.language}
    - Forbidden Words: {', '.join(rules.forbidden_words)}
    - DNA: {rules.brand_dna}
    - Formats: {rules.formatting_rules}
    """

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

    try:
        result = await run_optimizer_with_fallback(prompt)
        final_data = getattr(result, "data", getattr(result, "output", result))
        return {"final_proposal": final_data}
    except Exception as e:
        error_message = _format_error_for_log(e)
        print(f"❌ [Nodo 3] Error de IA: {error_message}")
        return {"error": error_message}


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
    prompt = f"""
    BRAND RULES TO ENFORCE:
    - Tone: {rules.tone_voice}
    - Forbidden Words: {', '.join(rules.forbidden_words)}
    - Brand DNA: {rules.brand_dna}

    AI PROPOSAL TO REVIEW:
    {proposal}

    Verify if the proposal follows all rules strictly.
    If it uses ANY forbidden words, reject it (is_perfect=False) and list them.
    If the title is over 70 chars, reject it.
    Provide actionable feedback for the writer to fix it.
    """

    try:
        result = await run_critic_with_fallback(prompt)
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
    print("💾 [Nodo 5] Guardando propuesta aprobada en Supabase...")
    proposal = state.get("final_proposal")

    if state.get("error"):
        return {}
    if not proposal:
        return {"error": "No hay propuesta aprobada para guardar."}

    try:
        proposal_dict = _proposal_to_dict(proposal)
        score = proposal_dict.get("audit_score", 80)
        audit_log_data = proposal_dict.get("audit_log", [])

        await _run_sync(lambda: supabase.table("shopify_products").update({
            "ai_proposal": proposal_dict,
            "audit_score": score,
            "audit_log": audit_log_data,
            "audit_status": STATUS_READY_TO_PUBLISH,
            "error_log": None,
            "retry_attempts": 0,
            "next_retry_at": None,
        }).eq("id", state["product_id"]).execute())

        print("✅ [Nodo 5] Producto marcado como READY_TO_PUBLISH.")
        if not state.get("auto_pilot_enabled", False):
            user_id = state.get("user_id")
            if user_id:
                await _run_sync(
                    lambda: supabase.rpc("increment_profile_credits_used", {"p_user_id": user_id}).execute()
                )
                print(f"💳 [Créditos] Crédito consumido para usuario {user_id} (READY_TO_PUBLISH).")
            else:
                print("⚠️ [Créditos] No se cobró crédito: user_id ausente.")

        return {"status": STATUS_READY_TO_PUBLISH, "retry_attempts": 0}
    except Exception as e:
        error_message = _format_error_for_log(e)
        print(f"❌ [Nodo 5] Error guardando propuesta: {error_message}")
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
        update_data: dict[str, Any] = {
            "audit_status": STATUS_NEEDS_OPTIMIZATION,
            "error_log": error_log,
        }

        if proposal:
            proposal_dict = _proposal_to_dict(proposal)
            update_data["ai_proposal"] = proposal_dict
            update_data["audit_score"] = proposal_dict.get("audit_score", 0)
            update_data["audit_log"] = proposal_dict.get("audit_log", [])

        await _run_sync(
            lambda: supabase.table("shopify_products")
            .update(update_data)
            .eq("id", product_id)
            .execute()
        )
        print(f"🟠 [Nodo 5B] Producto {product_id} marcado como NEEDS_OPTIMIZATION.")
        return {"status": STATUS_NEEDS_OPTIMIZATION}
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
            .single()
            .execute()
        )
        integration_data = integration_res.data or {}
        if not integration_data:
            raise ValueError(f"No hay integración Shopify para usuario {user_id}")

        await publish_product_to_shopify(
            shop_url=integration_data.get("shop_url", ""),
            access_token=integration_data.get("access_token", ""),
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

        await _run_sync(
            lambda: supabase.rpc("increment_profile_credits_used", {"p_user_id": user_id}).execute()
        )
        print(f"💳 [Créditos] Crédito consumido para usuario {user_id} (OPTIMIZED).")
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
    product_id = str(state["product_id"])
    fallback_message = (
        "El Juez no devolvió feedback legible."
        if state.get("critic_feedback") is None
        else "Error desconocido en LangGraph."
    )
    error_message = str(state.get("error") or fallback_message)
    print(f"🧯 [Error Handler] Marcando producto {product_id} como ERROR: {error_message}")
    await mark_product_error(product_id, error_message, state)
    return {"status": STATUS_ERROR}


def route_after_start(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "fetch_data"


def route_after_fetch(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "memory"


def route_after_memory(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "retrieve_knowledge"


def route_after_knowledge(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "ai_writer"


def route_after_save(state: KatalogState) -> str:
    if state.get("error"):
        return "error_handler"
    if state.get("auto_pilot_enabled", False):
        return "publish_to_shopify"
    return "end"


def route_after_publish(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "end"


def route_after_needs_optimization(state: KatalogState) -> str:
    return "error_handler" if state.get("error") else "end"


def build_graph():
    workflow = StateGraph(KatalogState)

    workflow.add_node("start_processing", start_processing)
    workflow.add_node("fetch_data", fetch_db_data)
    workflow.add_node("memory", retrieve_memory_letta)
    workflow.add_node("retrieve_knowledge", retrieve_knowledge)
    workflow.add_node("ai_writer", audit_and_write_pydantic)
    workflow.add_node("critic", review_proposal)
    workflow.add_node("save_db", save_to_supabase)
    workflow.add_node("needs_optimization", mark_needs_optimization)
    workflow.add_node("publish_to_shopify", publish_to_shopify_node)
    workflow.add_node("error_handler", error_handler)

    workflow.set_entry_point("start_processing")

    workflow.add_conditional_edges(
        "start_processing",
        route_after_start,
        {"fetch_data": "fetch_data", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "fetch_data",
        route_after_fetch,
        {"memory": "memory", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "memory",
        route_after_memory,
        {"retrieve_knowledge": "retrieve_knowledge", "error_handler": "error_handler"},
    )
    workflow.add_conditional_edges(
        "retrieve_knowledge",
        route_after_knowledge,
        {"ai_writer": "ai_writer", "error_handler": "error_handler"},
    )
    workflow.add_edge("ai_writer", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "save_db": "save_db",
            "ai_writer": "ai_writer",
            "needs_optimization": "needs_optimization",
            "error_handler": "error_handler",
        },
    )
    workflow.add_conditional_edges(
        "save_db",
        route_after_save,
        {
            "publish_to_shopify": "publish_to_shopify",
            "error_handler": "error_handler",
            "end": END,
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
