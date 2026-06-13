"""
agents/orchestrator_agent.py
El Director de Estrategia de Katalog AI.

NO escribe copy.
NO busca en internet.
SOLO diagnostica y diseña el Plan de Vuelo.

TIPOS DE DATOS SUPABASE (verificados via MCP):
  shopify_products.id   → BIGINT (int en Python)
  shopify_products.user_id → UUID (str en Python)
  merchant_alerts.product_id → BIGINT (int en Python)
"""

import logging
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from core.schemas import OrchestratorPlan

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorDeps:
    """Contexto completo del producto para el Orquestador."""
    product: dict                    # datos de Shopify / shopify_products
    metrics: dict                    # orders_count_7d, orders_count_30d, etc.
    cached_specs: dict | None        # specs del caché product_enrichment (puede ser None)
    precio_relativo: float           # precio / avg_categoria
    product_type_class: str          # MANUFACTURED | ARTISANAL | GENERIC
    available_skills: list[str]      # nombres de skills disponibles en skills/
    seo_score_category: str          # POOR | ACCEPTABLE | GOOD | EXCELLENT
    seo_score_raw: int               # 0-100


orchestrator_agent = Agent(
    model="google:gemini-3.5-flash",   # prefijo correcto — NO google-gla:
    output_type=OrchestratorPlan,       # NO result_type
    deps_type=OrchestratorDeps,
    retries=3,                          # reintentos automáticos si JSON inválido
    system_prompt="""
Eres el Director de Marketing B2B de Katalog AI.
Tu única función es DIAGNOSTICAR el problema de un producto de Shopify
y DISEÑAR un Plan de Vuelo (JSON) para el equipo de subagentes.

NO escribas copy. NO busques en internet. SOLO diagnostica y planifica.

═══════════════════════════════════════════════════════
LEY ABSOLUTA — FACT-ANCHORED VALUE (RIESGO LEGAL)
═══════════════════════════════════════════════════════
Si precio_relativo > 1.5 (producto premium), TIENES PROHIBIDO
inventar justificaciones de precio.

PROCESO OBLIGATORIO:
1. Revisa el product_dossier (cached_specs).
2. ¿Tiene material verificado? ¿Garantía real? ¿Certificación?
   → SÍ: usa esos hechos en copywriter_instructions.
   → NO: marca fact_anchored_alert=True y genera merchant_alert_message
         explicando qué datos debe agregar el comerciante en Shopify.
3. Si el dossier está VACÍO: NO des instrucciones de precio premium al
   Redactor. El copy debe ser sobrio e informativo.

NUNCA inventes: garantías, materiales, certificaciones, años de vida útil.
Un copy falso genera disputas de Stripe y demandas por fraude.

═══════════════════════════════════════════════════════
CÓMO DECIDIR SI ACTIVAR AL INVESTIGADOR WEB
═══════════════════════════════════════════════════════
activate_researcher_agent = True SOLO si:
1. product_type_class = "MANUFACTURED" O "GENERIC"
2. Y cached_specs está vacío o incompleto
3. Y hay gaps técnicos críticos (voltaje, material, certificación, dimensiones)

activate_researcher_agent = False si:
- cached_specs ya tiene las specs principales
- product_type_class = "ARTISANAL" (no hay specs que buscar)
- El copy_score es GOOD o EXCELLENT (el copy ya está bien)

═══════════════════════════════════════════════════════
INSTRUCCIONES PARA LOS SUBAGENTES
═══════════════════════════════════════════════════════
copywriter_instructions: Sé QUIRÚRGICO y ESPECÍFICO.
  Menciona: el problema principal a resolver, el tono correcto,
  qué framework usar (FAB/PAS), cómo justificar el precio (solo con hechos),
  qué skills del menú aplicar y POR QUÉ.
  MAL: "Escribe un copy persuasivo con Cialdini."
  BIEN: "El producto cuesta $400 (3x el promedio). El dossier confirma
         motor brushless y garantía 2 años. Usa ogilvy_price_anchoring
         para el H2, lista 3 specs técnicas del dossier como bullets FAB,
         y resuelve la objeción de durabilidad en el primer párrafo."

judge_instructions: Qué debe verificar el Juez específicamente.
  Menciona: qué afirmaciones técnicas comprobar contra el dossier,
  si el precio fue justificado con hechos reales, qué frases prohibidas
  detectar (garantías inventadas, materiales no verificados).
""",
)


@orchestrator_agent.system_prompt
def inject_product_context(ctx: RunContext[OrchestratorDeps]) -> str:
    """Inyecta el contexto completo del producto en el system prompt."""
    d = ctx.deps

    specs_text = "VACÍO — no hay specs verificadas" if not d.cached_specs else \
        "\n".join(f"  - {k}: {v}" for k, v in d.cached_specs.items())

    # Métricas reales de la DB: orders_count_7d, orders_count_30d
    orders_7d  = d.metrics.get("orders_count_7d",  d.metrics.get("sales_7d",  0))
    orders_30d = d.metrics.get("orders_count_30d", d.metrics.get("sales_30d", 0))
    orders_14d = d.metrics.get("orders_count_14d", 0)
    avg_weekly = round(orders_30d / 4.3, 1)  # aproximado a 4.3 semanas/mes

    return f"""
═══════════════ EXPEDIENTE DEL PRODUCTO ═══════════════

PRODUCTO:
  Título:       {d.product.get('current_title', d.product.get('title', 'Sin título'))}
  Vendor:       {d.product.get('vendor', 'Sin vendor')}
  Precio:       ${d.product.get('price', 0)}
  Tags:         {d.product.get('tags', 'Sin tags')}
  Descripción actual ({len(d.product.get('current_body_html', d.product.get('descriptionHtml', '')))} chars):
    {(d.product.get('current_body_html', d.product.get('descriptionHtml', '')))[:300]}...

CLASIFICACIÓN DEL PRODUCTO: {d.product_type_class}

CALIDAD DEL COPY ACTUAL:
  SEO Score: {d.seo_score_raw}/100 → {d.seo_score_category}

MÉTRICAS DE VENTAS:
  Últimos 7 días:  {orders_7d} pedidos
  Últimos 14 días: {orders_14d} pedidos
  Últimos 30 días: {orders_30d} pedidos
  Promedio semanal (30d): {avg_weekly} pedidos/semana

ANÁLISIS DE PRECIO:
  Precio relativo vs nicho: {d.precio_relativo}x
  {"⚠️ PRODUCTO PREMIUM — requiere Fact-Anchored Value" if d.precio_relativo > 1.5 else "✅ Precio normal para el nicho"}

SPECS VERIFICADAS (Dossier):
{specs_text}

SKILLS DISPONIBLES EN LA BIBLIOTECA:
{', '.join(d.available_skills) if d.available_skills else 'Ninguna disponible'}

═══════════════════════════════════════════════════════
"""
