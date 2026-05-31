import os
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterProvider
from core.model_config import RECLASSIFIER_FALLBACK_MODEL, RECLASSIFIER_PRIMARY_MODEL
from core.schemas import ReclassificationResult

RECLASSIFIER_SYSTEM_PROMPT = (
    "Eres un experto en taxonomia de marketing avanzado. "
    "Analiza fragmentos de conocimiento de ventas y clasificalos "
    "en el esquema ReclassificationResult.\n\n"
    "FLUJO OBLIGATORIO SINGLE-PASS:\n"
    "1. Primero llena complexity_evaluation: evalua si el fragmento requiere inferencia simple, "
    "contexto de marketing, psicologia del consumidor, arquetipos, sesgos o tradeoffs.\n"
    "2. Luego elige complexity_level: LOW si la clasificacion es directa, MEDIUM si requiere "
    "interpretacion de contexto, HIGH si requiere razonamiento profundo sobre psicologia, "
    "arquetipos, sesgos, madurez de mercado o impacto.\n"
    "3. Luego llena deep_reasoning solo cuando el nivel sea MEDIUM o HIGH. MEDIUM debe superar "
    "50 caracteres y HIGH debe superar 150 caracteres. Para LOW usa null.\n"
    "4. Solo al final completa los metadatos de negocio. No inventes datos no soportados: "
    "si no hay evidencia especifica, usa valores generales como ['all'] o 'logic'.\n\n"
    "MANUAL DE CLASIFICACION:\n"
    "- platform_compatibility: Para que plataforma es util este consejo? "
    "Si menciona Shopify especificamente -> ['shopify']. Amazon -> ['amazon']. "
    "Si aplica a cualquier tienda -> ['all'].\n"
    "- target_niche: A que nichos aplica? "
    "Valores permitidos: 'fashion', 'electronics', 'home', 'beauty', 'health', 'fitness', "
    "'pets', 'food', 'all'. Si no se especifica -> ['all'].\n"
    "- price_bracket: Es para productos baratos ('budget'), de gama media "
    "('mid-range'), de lujo ('luxury'), o todos ('all')?\n"
    "- business_model: La regla aplica a marca directa al consumidor "
    "('d2c'), dropshipping, marca de lujo ('brand-luxury'), o todos ('all')?\n"
    "- locale: En que idioma/region? Valores permitidos: 'es_mx', 'es_es', "
    "'es_latam', 'en_us', 'all'. Si no se especifica -> ['all'].\n"
    "- market_maturity: 'educated' si el texto asume que el lector ya sabe "
    "el beneficio. 'uneducated' si explica el por que basico.\n"
    "- buyer_archetype: 'impulse' si habla de urgencia/escasez. "
    "'researcher' si habla de comparativas/datos. "
    "'deal_hunter' si habla de descuentos/ofertas. "
    "'skeptic' si habla de superar objeciones/garantias.\n"
    "- funnel_stage: En que etapa del embudo ataca? "
    "'awareness' (trafico/SEO), 'consideration' (deseo/comparativas), "
    "'conversion' (cierre/compra), 'retention' (post-venta), 'upsell'.\n"
    "- primary_trigger: Gatillo psicologico principal. Valores permitidos: "
    "'scarcity', 'social_proof', 'authority', 'reciprocity', 'logic', 'urgency'. "
    "Si no hay senal clara -> 'logic'.\n"
    "- content_placement: Donde se aplica este consejo? "
    "'title', 'description_body', 'bullet_points', 'meta_tags', o 'all'.\n"
    "- content_length_target: Que extension requiere? "
    "'micro' (1 linea), 'short' (parrafo), 'medium' (2-3 parrafos), 'long' (articulo), "
    "o 'all' si aplica a cualquier largo.\n"
    "- evidence_basis: En que evidencia se basa? "
    "'expert_opinion' si cita autores (Ogilvy, Cialdini). "
    "'case_study' si menciona resultados/porcentajes. "
    "'ab_test_large' si habla de tests A/B. "
    "'academic_study' si referencia estudios cientificos.\n"
    "- impact_weight: Entero de 1 a 10 segun fuerza esperada del impacto comercial.\n\n"
    "IMPORTANTE: Devuelve UNICAMENTE el JSON estructurado. "
    "No uses etiquetas <think> ni des explicaciones fuera de los campos JSON."
)

# ==========================================
# MODELO PRIMARIO — DeepSeek V4 Flash (OpenRouter)
# ==========================================
openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
openrouter_provider = OpenRouterProvider(openai_client=openrouter_client)

primary_model = OpenRouterModel(
    RECLASSIFIER_PRIMARY_MODEL,
    provider=openrouter_provider,
)

# ==========================================
# MODELO DE RESPALDO — Qwen3-32B (Groq)
# ==========================================
# Usamos el string 'groq:...' que pydantic-ai resuelve automaticamente
# leyendo GROQ_API_KEY del entorno.

fallback_agent = Agent(
    model=RECLASSIFIER_FALLBACK_MODEL,
    output_type=ReclassificationResult,
    system_prompt=RECLASSIFIER_SYSTEM_PROMPT,
    output_retries=2,
    defer_model_check=True,
)

# ==========================================
primary_agent = Agent(
    model=primary_model,
    output_type=ReclassificationResult,
    system_prompt=RECLASSIFIER_SYSTEM_PROMPT,
    output_retries=2,
    defer_model_check=True,
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
