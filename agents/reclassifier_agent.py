import os
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterProvider
from core.schemas import ReclassificationResult
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# MODELO PRIMARIO — DeepSeek V4 Flash (OpenRouter)
# ==========================================
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
openrouter_provider = OpenRouterProvider(openai_client=openrouter_client)

primary_model = OpenRouterModel(
    'deepseek/deepseek-v4-flash:free',
    provider=openrouter_provider,
)

# ==========================================
# MODELO DE RESPALDO — Qwen3-32B (Groq)
# ==========================================
# Usamos el string 'groq:...' que pydantic-ai resuelve automaticamente
# leyendo GROQ_API_KEY del entorno.

fallback_agent = Agent(
    model='groq:qwen/qwen3-32b',
    output_type=ReclassificationResult,
    system_prompt=(
        "Eres un experto en taxonomia de marketing avanzado. "
        "Analiza fragmentos de conocimiento de ventas y clasificalos "
        "en el esquema ReclassificationResult.\n\n"
        "MANUAL DE CLASIFICACION:\n"
        "- platform_compatibility: ?Para que plataforma es util este consejo? "
        "Si menciona Shopify especificamente -> ['shopify']. Amazon -> ['amazon']. "
        "Si aplica a cualquier tienda -> ['all'].\n"
        "- target_niche: ?A que nichos aplica? "
        "Si menciona 'fashion', 'electronics', etc, usalos. Si no -> ['all'].\n"
        "- price_bracket: ?Es para productos baratos ('budget'), de gama media "
        "('mid-range'), de lujo ('luxury'), o todos ('all')?\n"
        "- business_model: ?La regla aplica a marca directa al consumidor "
        "('d2c'), dropshipping, marca de lujo, o todos ('all')?\n"
        "- locale: ?En que idioma/region? Si no se especifica -> ['all'].\n"
        "- market_maturity: 'educated' si el texto asume que el lector ya sabe "
        "el beneficio. 'uneducated' si explica el 'por que' basico.\n"
        "- buyer_archetype: 'impulse' si habla de urgencia/escasez. "
        "'researcher' si habla de comparativas/datos. "
        "'deal_hunter' si habla de descuentos/ofertas. "
        "'skeptic' si habla de superar objeciones/garantias.\n"
        "- funnel_stage: ?En que etapa del embudo ataca? "
        "'awareness' (trafico/SEO), 'consideration' (deseo/comparativas), "
        "'conversion' (cierre/compra), 'retention' (post-venta), 'upsell'.\n"
        "- primary_trigger: ?Cual es el gatillo psicologico principal? "
        "Ej: 'scarcity', 'social_proof', 'authority', 'reciprocity', 'logic', 'urgency'.\n"
        "- content_placement: ?Donde se aplica este consejo? "
        "'title', 'description_body', 'bullet_points', 'meta_tags', o 'all'.\n"
        "- content_length_target: ?Que extension requiere? "
        "'micro' (1 linea), 'short' (parrafo), 'medium' (2-3 parrafos), 'long' (articulo).\n"
        "- evidence_basis: ?En que evidencia se basa? "
        "'expert_opinion' si cita autores (Ogilvy, Cialdini). "
        "'case_study' si menciona resultados/porcentajes. "
        "'ab_test_large' si habla de tests A/B. "
        "'academic_study' si referencia estudios cientificos.\n\n"
        "IMPORTANTE: Devuelve UNICAMENTE el JSON estructurado. "
        "No uses etiquetas <think> ni des explicaciones."
    )
)

# ==========================================
primary_agent = Agent(
    model=primary_model,
    output_type=ReclassificationResult,
    system_prompt=(
        "Eres un experto en taxonomia de marketing avanzado. "
        "Analiza fragmentos de conocimiento de ventas y clasificalos "
        "en el esquema ReclassificationResult.\n\n"
        "MANUAL DE CLASIFICACION:\n"
        "- platform_compatibility: ?Para que plataforma es util este consejo? "
        "Si menciona Shopify especificamente -> ['shopify']. Amazon -> ['amazon']. "
        "Si aplica a cualquier tienda -> ['all'].\n"
        "- target_niche: ?A que nichos aplica? "
        "Si menciona 'fashion', 'electronics', etc, usalos. Si no -> ['all'].\n"
        "- price_bracket: ?Es para productos baratos ('budget'), de gama media "
        "('mid-range'), de lujo ('luxury'), o todos ('all')?\n"
        "- business_model: ?La regla aplica a marca directa al consumidor "
        "('d2c'), dropshipping, marca de lujo, o todos ('all')?\n"
        "- locale: ?En que idioma/region? Si no se especifica -> ['all'].\n"
        "- market_maturity: 'educated' si el texto asume que el lector ya sabe "
        "el beneficio. 'uneducated' si explica el 'por que' basico.\n"
        "- buyer_archetype: 'impulse' si habla de urgencia/escasez. "
        "'researcher' si habla de comparativas/datos. "
        "'deal_hunter' si habla de descuentos/ofertas. "
        "'skeptic' si habla de superar objeciones/garantias.\n"
        "- funnel_stage: ?En que etapa del embudo ataca? "
        "'awareness' (trafico/SEO), 'consideration' (deseo/comparativas), "
        "'conversion' (cierre/compra), 'retention' (post-venta), 'upsell'.\n"
        "- primary_trigger: ?Cual es el gatillo psicologico principal? "
        "Ej: 'scarcity', 'social_proof', 'authority', 'reciprocity', 'logic', 'urgency'.\n"
        "- content_placement: ?Donde se aplica este consejo? "
        "'title', 'description_body', 'bullet_points', 'meta_tags', o 'all'.\n"
        "- content_length_target: ?Que extension requiere? "
        "'micro' (1 linea), 'short' (parrafo), 'medium' (2-3 parrafos), 'long' (articulo).\n"
        "- evidence_basis: ?En que evidencia se basa? "
        "'expert_opinion' si cita autores (Ogilvy, Cialdini). "
        "'case_study' si menciona resultados/porcentajes. "
        "'ab_test_large' si habla de tests A/B. "
        "'academic_study' si referencia estudios cientificos.\n\n"
        "IMPORTANTE: Devuelve UNICAMENTE el JSON estructurado. "
        "No uses etiquetas <think> ni des explicaciones."
    )
)

reclassifier_agent = primary_agent


async def classify_with_smart_fallback(content: str) -> ReclassificationResult:
    """Intento 1: DeepSeek V4 Flash. Fallback: Qwen3-32B (Groq) si hay error 429/404/503."""
    try:
        result = await primary_agent.run(content)
        return getattr(result, "data", getattr(result, "output", result))
    except ModelHTTPError as e:
        if e.status_code in (429, 404, 503):
            print(
                f"\u26a0\ufe0f [Fallback] OpenRouter/DeepSeek fuera de servicio "
                f"({e.status_code}). Activando Qwen3-32B (Razonamiento Profundo)..."
            )
            result = await fallback_agent.run(content)
            return getattr(result, "data", getattr(result, "output", result))
        raise
