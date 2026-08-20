from typing import Any, NotRequired, Required, TypedDict
from core.schemas import ProductContext, BrandRules, AIProposalOutput, CriticFeedback, OrchestratorPlan

class KatalogState(TypedDict, total=False):
    """
    La memoria a corto plazo (RAM) de la ejecución actual.
    Ahora con 'critic_feedback' e 'iterations' para soportar el bucle de calidad.

    TIPOS DE DATOS SUPABASE (verificados via MCP pre-flight check):
      shopify_products.id   → BIGINT → product_id es int en Python
      shopify_products.user_id → UUID → user_id es str en Python
    """
    # product_id es int porque shopify_products.id es BIGINT en Supabase
    product_id: Required[int]
    user_id: NotRequired[str]
    auto_pilot_enabled: NotRequired[bool]
    current_status: NotRequired[str]
    retry_attempts: NotRequired[int]
    # Intentos de publicación ya fallidos (shopify_products.publish_attempts).
    # Lo carga fetch_db_data; el Nodo 6 lo usa para backoff y tope de reintentos.
    publish_attempts: NotRequired[int]
    product_context: NotRequired[ProductContext]
    brand_rules: NotRequired[BrandRules]

    # Memoria y Propuestas
    letta_memory: NotRequired[str]
    final_proposal: NotRequired[AIProposalOutput | dict[str, Any]]

    # RAG - Knowledge Base Retrieval
    rag_knowledge: NotRequired[list[dict[str, Any]]]

    # 🛡️ CAPAS DE SEGURIDAD Y BUCLE DE CRÍTICA
    critic_feedback: NotRequired[CriticFeedback | Any]  # Aquí vive el veredicto del Juez
    iterations: NotRequired[int]                        # Contador de seguridad (Max 3 intentos)

    # ─── RESERVE → COMMIT (BILLING) ──────────────────────────────────────────
    reservation_id: NotRequired[str]       # UUID de la reserva activa en Supabase
    credits_reserved: NotRequired[int]     # Créditos reservados (siempre 1 en el modelo actual)
    writer_invoked: NotRequired[bool]      # True una vez que ai_writer corrió al menos una vez
    out_of_credits: NotRequired[bool]      # True si start_processing detectó falta de créditos

    # Manejo de errores global
    error: NotRequired[str]
    status: NotRequired[str]

    # Metadatos de optimización
    framework_used: NotRequired[str]
    tone_used: NotRequired[str]
    description_length: NotRequired[int]

    # Taxonomía Predictiva de Shopify
    taxonomy_context: NotRequired[str]             # prosa lista para el prompt (categoría ± atributos)
    taxonomy_available: NotRequired[bool]          # api_ok de la consulta (False = falló la API)
    taxonomy_attributes: NotRequired[list[str]]    # atributos CON valores verificados; vacío hoy (iteración 2)

    # ─── CAPA DEL ORQUESTADOR (NUEVOS CAMPOS) ────────────────────────────────
    # Clasificación determinista del producto (sin LLM)
    product_type_class: NotRequired[str]           # MANUFACTURED | ARTISANAL | GENERIC
    product_quadrant: NotRequired[str]             # NEEDS_OPT | STABLE | MONITORING | BENCHMARK | INVESTIGATE_CAUSE
    precio_relativo: NotRequired[float]            # precio / avg_precio_nicho
    seo_score_raw: NotRequired[int]                # 0-100 (de shopify_products.seo_score_initial)
    seo_score_category: NotRequired[str]           # POOR | ACCEPTABLE | GOOD | EXCELLENT
    cached_specs: NotRequired[dict | None]         # specs del caché product_enrichment
    # Hechos NIVEL 1: metafields estructurados de Shopify, verificados por
    # definicion (los emite la plataforma). Distintos de cached_specs, que
    # viene de busqueda web y es NIVEL 3/4.
    verified_facts: NotRequired[list[str]]
    data_gaps: NotRequired[list[str]]              # gaps detectados (para el Investigador)
    available_skills: NotRequired[list[str]]       # nombres de skills disponibles
    fingerprint: NotRequired[str]                  # SHA-256 del producto base

    # Plan de vuelo del Orquestador
    orchestrator_plan: NotRequired[OrchestratorPlan | None]

    # Resultado del Investigador (si fue activado)
    research_result: NotRequired[Any | None]       # ResearchResult de researcher_agent

    # Skills pre-cargadas (instrucciones concatenadas para el Redactor)
    loaded_skills: NotRequired[str]

    # Estado de monitoreo
    monitoring_since: NotRequired[str | None]      # ISO timestamp (TIMESTAMPTZ en DB)

    # Producto raw de Shopify (dict con todos los campos de shopify_products)
    product: NotRequired[dict[str, Any]]

    # Métricas de ventas (de product_metrics)
    metrics: NotRequired[dict[str, Any]]
