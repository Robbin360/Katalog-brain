"""
agents/researcher_agent.py
==========================
Agente Investigador de Katalog AI — Stack PydanticAI v2 nativo.
 
Características:
- PydanticAI Agent con output_type estructurado (sin json.loads manual)
- Reintentos automáticos (retries=3) cuando la IA se equivoca en formato
- Asincronía pura (sin run_in_executor ni SDK bloqueante)
- Query expansion: 5 ángulos de búsqueda por producto
- Búsqueda en paralelo con asyncio.Semaphore
- Clasificación de fuentes: fabricante > distribuidor > secundaria
- Síntesis multi-fuente con detección de contradicciones
- Decisión de cuándo seguir buscando (máx. 3 rondas)
- Caché SHA-256 permanente en Supabase (product_enrichment)
 
Dependencias:
    pip install pydantic-ai httpx supabase
"""
 
# ─── IMPORTS ─────────────────────────────────────────────────────────────────
 
import asyncio
import hashlib
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
 
import httpx
from pydantic import BaseModel
from pydantic_ai import Agent
from supabase import create_client, Client
from core.model_config import RESEARCHER_MODEL
 
logger = logging.getLogger(__name__)
 
# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
 
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY", "")
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY", "")
SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY      = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
 
# PydanticAI lee GOOGLE_API_KEY del entorno automáticamente
# No se necesita inicialización manual del cliente Gemini
 
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
 
GEMINI_MODEL = "google:gemini-3.5-flash"   # modelo Flash 2026 (prefijo actual de PydanticAI)
 
# ─── ENUMS ────────────────────────────────────────────────────────────────────
 
class ConfidenceLevel(str, Enum):
    HIGH   = "high"    # fabricante oficial
    MEDIUM = "medium"  # distribuidor confiable
    LOW    = "low"     # fuente secundaria
 
 
class SourceType(str, Enum):
    MANUFACTURER = "manufacturer"
    DISTRIBUTOR  = "distributor"
    SECONDARY    = "secondary"
 
 
# ─── MODELOS PYDANTIC — SALIDAS ESTRUCTURADAS DE LA IA ───────────────────────
 
class SearchQuery(BaseModel):
    """Una query individual generada por el expander."""
    query:    str
    angle:    str  # official_site | direct_spec | distributor | series_broad | alternative
    priority: int  # 1 = más importante
 
 
class QueryExpansionResult(BaseModel):
    """
    Salida estructurada del expander_agent.
    PydanticAI valida y reintenta si la IA no respeta este schema.
    """
    queries: list[SearchQuery]
 
 
class FoundData(BaseModel):
    """Un dato encontrado con metadatos de confianza."""
    value:                  str
    confirmed_by_sources:   int   = 1
    contradiction_detected: bool  = False
    alternative_value:      Optional[str] = None
 
 
class SynthesisResult(BaseModel):
    """
    Salida estructurada del synthesizer_agent.
    PydanticAI valida y reintenta si la IA no respeta este schema.
    """
    found:          dict[str, FoundData]
    contradictions: dict[str, dict]   # {campo: {source_1_value, source_2_value}}
    not_found:      list[str]
 
 
# ─── MODELOS INTERNOS ─────────────────────────────────────────────────────────
 
class DataPoint(BaseModel):
    """Un dato verificado listo para el Redactor."""
    value:                  str
    confidence:             ConfidenceLevel
    source_url:             str
    source_type:            str
    contradiction_detected: bool = False
    alternative_value:      Optional[str] = None
 
 
class ResearchResult(BaseModel):
    """Resultado final del Agente Investigador."""
    found_data:        dict[str, DataPoint]
    not_found:         list[str]
    source_url:        str
    raw_content:       str
    search_query_used: str
    total_cost_usd:    float
    rounds_used:       int
    from_cache:        bool = False
 
 
@dataclass
class SearchAttempt:
    """Resultado de una búsqueda individual."""
    query:       str
    angle:       str
    priority:    int
    results:     list[dict] = field(default_factory=list)
    best_url:    str = ""
    raw_content: str = ""
    confidence:  str = "low"
    source_type: str = "secondary"
    error:       str = ""
 
 
# ─── AGENTES PYDANTIC AI ─────────────────────────────────────────────────────
 
expander_agent = Agent(
    model=RESEARCHER_MODEL,
    output_type=QueryExpansionResult,
    retries=3,  # reintenta automáticamente si el JSON no es válido
    system_prompt="""
Eres un experto en búsqueda técnica industrial y de e-commerce.
Tu trabajo es generar queries de búsqueda diversas para encontrar
especificaciones técnicas de productos.
 
REGLAS ESTRICTAS:
1. Genera exactamente 5 queries con ángulos distintos
2. Cada query debe atacar el problema desde una perspectiva diferente
3. Prioridad 1 = más importante, 5 = menos importante
4. Las queries deben ser específicas y buscables en Google
5. Incluye siempre el modelo exacto del producto cuando esté disponible
""",
)
 
synthesizer_agent = Agent(
    model=RESEARCHER_MODEL,
    output_type=SynthesisResult,
    retries=3,  # reintenta si no respeta el schema
    system_prompt="""
Eres un extractor de datos técnicos de alta precisión.
Tu trabajo es extraer datos específicos de múltiples fuentes web
y detectar contradicciones entre ellas.
 
REGLAS ESTRICTAS:
1. Extrae SOLO datos que aparezcan EXPLÍCITAMENTE en el texto
2. Copia los valores EXACTAMENTE como aparecen, sin parafrasear
3. Si el mismo dato aparece en múltiples fuentes, incrementa confirmed_by_sources
4. Si dos fuentes dan valores DISTINTOS, marca contradiction_detected=true
5. Si un dato NO aparece en ninguna fuente, inclúyelo en not_found
6. NUNCA inventes ni inferras valores
""",
)
 
 
# ─── CACHÉ SHA-256 ────────────────────────────────────────────────────────────
 
VARIANT_PATTERNS = [
    r'\b(negro|blanco|rojo|azul|verde|gris|beige|dorado|plateado|rosa|morado)\b',
    r'\b(black|white|red|blue|green|gray|grey|gold|silver|pink|purple|orange)\b',
    r'\b(xs|sm|md|lg|xl|xxl|xxxl|talla|size)\b',
    r'\b(3[6-9]|4[0-9]|5[0-2])\b',
    r'\b(nuevo|new|edicion|edition|especial|special|pack|kit|set|bundle)\b',
]
 
 
def build_fingerprint(vendor: str, product_type: str, title: str) -> str:
    """
    SHA-256 del modelo base eliminando variantes (color, talla).
    Garantiza que el caché funcione para todas las variantes del mismo modelo.
    """
    base = title.lower().strip()
    for pattern in VARIANT_PATTERNS:
        base = re.sub(pattern, '', base, flags=re.IGNORECASE)
    base = ' '.join(base.split())
 
    raw = f"{vendor.lower().strip()}_{product_type.lower().strip()}_{base}"
    return hashlib.sha256(raw.encode()).hexdigest()
 
 
async def check_enrichment_cache(fingerprint: str) -> Optional[dict]:
    """Busca en product_enrichment. Retorna datos si existe, None si no."""
    try:
        result = (
            supabase.table("product_enrichment")
            .select("technical_specs, raw_content, source_url")
            .eq("product_fingerprint", fingerprint)
            .execute()
        )
        if result.data:
            logger.info(f"Cache HIT: {fingerprint[:8]}...")
            return result.data[0]
        return None
    except Exception as e:
        logger.error(f"Error verificando caché: {e}")
        return None
 
 
async def save_to_enrichment_cache(
    fingerprint:     str,
    technical_specs: dict,
    raw_content:     str,
    source_url:      str,
    search_cost_usd: float,
) -> bool:
    """Guarda los datos en el caché permanente de Supabase."""
    try:
        supabase.table("product_enrichment").upsert(
            {
                "product_fingerprint": fingerprint,
                "technical_specs":     technical_specs,
                "raw_content":         raw_content[:10_000],
                "source_url":          source_url,
                "search_cost_usd":     search_cost_usd,
                "expires_at":          None,  # specs físicas no expiran
            },
            on_conflict="product_fingerprint",
        ).execute()
        logger.info(f"Cache guardado: {fingerprint[:8]}...")
        return True
    except Exception as e:
        logger.error(f"Error guardando caché: {e}")
        return False
 
 
# ─── DETECCIÓN DE GAPS ────────────────────────────────────────────────────────
 
TECHNICAL_SIGNALS = [
    r'[A-Z]{2,}\d{3,}',               # modelos alfanuméricos: 3RV1021
    r'\d+\s*(amp|volt|watt|hz|rpm)',    # unidades eléctricas
    r'\b(iec|ul |ce |iso|nema)\b',     # certificaciones
    r'\d+\s*(psi|mpa|bar|kpa)',        # unidades de presión
    r'\b\d{4,}\s*mah\b',               # baterías
]
 
 
def needs_web_research(product: dict) -> tuple[bool, str, list[str]]:
    """
    Determina si el producto necesita búsqueda web y qué gaps hay.
    Retorna: (necesita_buscar, razón, gaps_detectados)
    """
    title       = (product.get("title")           or "").strip()
    description = (product.get("descriptionHtml") or "").strip()
    vendor      = (product.get("vendor")          or "").strip()
    product_type = (product.get("productType")    or "").strip()
    tags        = product.get("tags")       or []
    metafields  = product.get("metafields") or {}
 
    score = 0
    gaps: list[str] = []
 
    if len(description) > 150:
        score += 3
    else:
        gaps.append("description")
 
    if metafields.get("material") or metafields.get("custom.material"):
        score += 2
    else:
        gaps.append("material")
 
    if len(tags) >= 3:
        score += 2
    else:
        gaps.append("tags")
 
    if product_type:
        score += 1
    else:
        gaps.append("product_type")
 
    if vendor:
        score += 1
 
    if metafields.get("weight") or metafields.get("dimensions"):
        score += 1
    else:
        gaps.append("dimensions")
 
    # Señales técnicas en el título → buscar siempre
    for signal in TECHNICAL_SIGNALS:
        if re.search(signal, title, re.IGNORECASE):
            return True, "producto_tecnico", gaps
 
    if len(description) < 50 and not metafields:
        return True, "sin_datos_basicos", gaps
 
    if score < 4:
        return True, "datos_insuficientes", gaps
 
    return False, "datos_suficientes", []
 
 
# ─── PASO 1: QUERY EXPANSION CON PYDANTIC AI ─────────────────────────────────
 
async def expand_queries(
    product:      dict,
    gaps:         list[str],
    instructions: str,
) -> list[dict]:
    """
    Genera 5 queries con ángulos distintos usando PydanticAI.
    La salida es validada automáticamente contra QueryExpansionResult.
    Si la IA falla el formato, PydanticAI reintenta hasta 3 veces.
    """
    vendor = product.get("vendor", "")
    title  = product.get("title", "")
 
    prompt = f"""
Producto: {vendor} {title}
Datos que necesito encontrar: {", ".join(gaps)}
Instrucciones adicionales: {instructions}
 
Genera exactamente 5 queries de búsqueda con estos ángulos:
 
1. OFFICIAL_SITE: modelo exacto + "datasheet" + site:fabricante.com
   Ejemplo: "Siemens 3RV1021-1JA10 datasheet site:siemens.com"
 
2. DIRECT_SPEC: modelo + nombre técnico del dato buscado
   Ejemplo: "3RV1021-1JA10 rated voltage IEC certification"
 
3. DISTRIBUTOR: modelo en distribuidores confiables
   Ejemplo: "3RV1021-1JA10 specifications site:grainger.com OR site:mouser.com"
 
4. SERIES_BROAD: familia del producto, no modelo exacto
   Ejemplo: "Siemens Sirius 3RV1 series technical specifications"
 
5. ALTERNATIVE: sinónimos o términos alternativos del sector
   Ejemplo: "Siemens Sirius motor starter protector 3RV specifications"
"""
 
    try:
        result = await expander_agent.run(prompt)
        return [q.model_dump() for q in result.output.queries]
    except Exception as e:
        logger.error(f"expand_queries falló: {e}")
        # Fallback: query básica
        return [{"query": f"{vendor} {title} specifications datasheet",
                 "angle": "basic", "priority": 1}]
 
 
# ─── CLASIFICACIÓN DE FUENTES ─────────────────────────────────────────────────
 
MANUFACTURERS = {
    "siemens.com", "bosch.com", "abb.com", "schneider-electric.com",
    "philips.com", "dewalt.com", "3m.com", "nike.com", "apple.com",
    "samsung.com", "lg.com", "panasonic.com", "sony.com", "canon.com",
    "adidas.com", "puma.com", "honeywell.com", "emerson.com",
}
 
DISTRIBUTORS = {
    "grainger.com", "mouser.com", "digikey.com", "rs-online.com",
    "amazon.com", "automation24.com", "farnell.com", "arrow.com",
    "newark.com", "mcmaster.com", "walmart.com", "homedepot.com",
}
 
 
def classify_source(url: str) -> tuple[str, ConfidenceLevel]:
    """Fabricante = HIGH, Distribuidor = MEDIUM, Otros = LOW."""
    url_lower = url.lower()
    if any(m in url_lower for m in MANUFACTURERS):
        return SourceType.MANUFACTURER, ConfidenceLevel.HIGH
    if any(d in url_lower for d in DISTRIBUTORS):
        return SourceType.DISTRIBUTOR, ConfidenceLevel.MEDIUM
    return SourceType.SECONDARY, ConfidenceLevel.LOW
 
 
def pick_best_url(results: list[dict]) -> Optional[dict]:
    """Elige el resultado con mayor confianza."""
    if not results:
        return None
 
    priority = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 2}
    classified = []
 
    for r in results:
        url = r.get("url", "")
        source_type, confidence = classify_source(url)
        classified.append({**r, "source_type": source_type,
                            "confidence": confidence,
                            "_sort": priority[confidence]})
 
    classified.sort(key=lambda x: x["_sort"])
    return classified[0]
 
 
# ─── PASO 3: TAVILY ───────────────────────────────────────────────────────────
 
async def search_with_tavily(query: str) -> list[dict]:
    """Búsqueda web. Costo: ~$0.008 por query."""
    if not TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY no configurada")
        return []
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key":      TAVILY_API_KEY,
                    "query":        query,
                    "search_depth": "advanced",
                    "max_results":  5,
                },
            )
        return response.json().get("results", []) if response.status_code == 200 else []
    except Exception as e:
        logger.error(f"Tavily error: {e}")
        return []
 
 
# ─── PASO 4: FIRECRAWL ────────────────────────────────────────────────────────
 
async def scrape_with_firecrawl(url: str) -> str:
    """Raspa contenido completo en Markdown. Costo: ~$0.005 por página."""
    if not FIRECRAWL_API_KEY:
        logger.warning("FIRECRAWL_API_KEY no configurada")
        return ""
    try:
        async with httpx.AsyncClient(timeout=25.0) as client:
            response = await client.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers={"Authorization": f"Bearer {FIRECRAWL_API_KEY}"},
                json={"url": url, "formats": ["markdown"],
                      "onlyMainContent": True, "waitFor": 2000},
            )
        if response.status_code == 200:
            return response.json().get("data", {}).get("markdown", "")
        return ""
    except Exception as e:
        logger.error(f"Firecrawl error: {e}")
        return ""
 
 
# ─── PASO 5: BÚSQUEDA EN PARALELO ────────────────────────────────────────────
 
async def search_single_query(attempt: SearchAttempt) -> SearchAttempt:
    """Una query: Tavily → mejor URL → Firecrawl (si confianza suficiente)."""
    try:
        results = await search_with_tavily(attempt.query)
        if not results:
            attempt.error = "sin_resultados"
            return attempt
 
        best = pick_best_url(results)
        if not best:
            attempt.error = "sin_url_valida"
            return attempt
 
        attempt.best_url    = best["url"]
        attempt.confidence  = best["confidence"]
        attempt.source_type = best["source_type"]
        attempt.results     = results
 
        # Firecrawl solo para fuentes de calidad suficiente
        if best["confidence"] in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM):
            attempt.raw_content = await scrape_with_firecrawl(best["url"])
 
        # Fallback al snippet de Tavily si Firecrawl falla o no se usó
        if not attempt.raw_content:
            attempt.raw_content = best.get("content", "")
 
    except Exception as e:
        attempt.error = str(e)
        logger.error(f"search_single_query error: {e}")
 
    return attempt
 
 
async def search_all_queries_parallel(
    queries:      list[dict],
    max_parallel: int = 3,
) -> list[SearchAttempt]:
    """
    Ejecuta todas las queries en paralelo.
    Semáforo limita concurrencia para respetar rate limits de las APIs.
    """
    semaphore = asyncio.Semaphore(max_parallel)
 
    async def bounded(q: dict) -> SearchAttempt:
        async with semaphore:
            attempt = SearchAttempt(
                query=q["query"], angle=q["angle"], priority=q["priority"]
            )
            return await search_single_query(attempt)
 
    results = await asyncio.gather(
        *[bounded(q) for q in queries], return_exceptions=True
    )
 
    valid = [r for r in results if isinstance(r, SearchAttempt)]
 
    priority_map = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 2}
    valid.sort(key=lambda x: (priority_map.get(x.confidence, 3), x.priority))
 
    logger.info(
        f"Paralelo: {len(valid)} exitosas, "
        f"{sum(1 for r in results if isinstance(r, Exception))} fallidas"
    )
    return valid
 
 
# ─── PASO 6: SÍNTESIS MULTI-FUENTE CON PYDANTIC AI ───────────────────────────
 
async def synthesize_multi_source(
    attempts: list[SearchAttempt],
    gaps:     list[str],
    product:  dict,
) -> tuple[dict[str, DataPoint], list[str]]:
    """
    Fusiona resultados de múltiples fuentes con PydanticAI.
    Detección automática de contradicciones.
    Salida validada contra SynthesisResult (sin json.loads manual).
    """
    valid = [a for a in attempts if a.raw_content and not a.error]
 
    if not valid:
        logger.warning("Sin fuentes válidas para síntesis")
        return {}, gaps
 
    # Construir contexto con todas las fuentes (máximo 4)
    sources_text = ""
    for i, a in enumerate(valid[:4]):
        sources_text += f"""
=== FUENTE {i + 1} ===
URL: {a.best_url}
Confianza: {a.confidence}
Tipo: {a.source_type}
Query usada: {a.query}
Contenido:
{a.raw_content[:2500]}
 
"""
 
    prompt = f"""
Producto buscado: {product.get("vendor", "")} {product.get("title", "")}
Datos a extraer: {", ".join(gaps)}
 
Fuentes disponibles:
{sources_text}
 
Extrae los datos solicitados cruzando todas las fuentes.
Marca contradiction_detected=true si dos fuentes dan valores distintos.
Incrementa confirmed_by_sources si múltiples fuentes confirman el mismo valor.
"""
 
    try:
        result = await synthesizer_agent.run(prompt)
        synthesis = result.output
 
        found_data: dict[str, DataPoint] = {}
        best_url    = valid[0].best_url if valid else ""
        best_stype  = valid[0].source_type if valid else "unknown"
 
        priority = {ConfidenceLevel.HIGH: 0, ConfidenceLevel.MEDIUM: 1, ConfidenceLevel.LOW: 2}
 
        for key, info in synthesis.found.items():
            # Confianza según cuántas fuentes confirmaron
            if info.confirmed_by_sources >= 3:
                conf = ConfidenceLevel.HIGH
            elif info.confirmed_by_sources == 2:
                conf = ConfidenceLevel.MEDIUM
            else:
                conf = valid[0].confidence if valid else ConfidenceLevel.LOW
 
            found_data[key] = DataPoint(
                value                  = info.value,
                confidence             = conf,
                source_url             = best_url,
                source_type            = best_stype,
                contradiction_detected = info.contradiction_detected,
                alternative_value      = info.alternative_value,
            )
 
        # Agregar contradicciones como DataPoints con advertencia
        for key, contra in synthesis.contradictions.items():
            if key not in found_data:
                found_data[key] = DataPoint(
                    value                  = contra.get("source_1_value", ""),
                    confidence             = ConfidenceLevel.LOW,
                    source_url             = best_url,
                    source_type            = "conflict",
                    contradiction_detected = True,
                    alternative_value      = contra.get("source_2_value"),
                )
 
        logger.info(
            f"Síntesis: {len(found_data)} encontrados, "
            f"{len(synthesis.not_found)} no encontrados"
        )
        return found_data, synthesis.not_found
 
    except Exception as e:
        logger.error(f"synthesize_multi_source error: {e}")
        return {}, gaps
 
 
# ─── PASO 7: DECISIÓN DE CONTINUAR ───────────────────────────────────────────
 
def should_search_more(
    found_data:     dict,
    not_found:      list[str],
    round_number:   int,
    critical_gaps:  list[str],
    max_rounds:     int = 3,
) -> tuple[bool, str]:
    """Decide si hacer otra ronda de búsqueda."""
    if round_number >= max_rounds:
        return False, f"limite_{max_rounds}_rondas"
 
    if not not_found:
        return False, "todos_encontrados"
 
    critical_missing = [g for g in not_found if g in critical_gaps]
    if not critical_missing:
        return False, "solo_faltan_opcionales"
 
    return True, f"criticos_faltantes: {critical_missing}"
 
 
# ─── FUNCIÓN PRINCIPAL ────────────────────────────────────────────────────────
 
async def research_product(
    product:               dict,
    research_instructions: str        = "",
    gaps_detected:         Optional[list[str]] = None,
    critical_gaps:         Optional[list[str]] = None,
    max_rounds:            int        = 3,
) -> ResearchResult:
    """
    Agente Investigador completo.
 
    Flujo:
    1. Verificar caché SHA-256 → si hit, retornar sin costo
    2. Expandir queries (5 ángulos) con PydanticAI
    3. Buscar en paralelo con Tavily + Firecrawl
    4. Sintetizar con PydanticAI (sin json.loads manual)
    5. Decidir si hacer más rondas
    6. Guardar en caché permanente
    """
    if gaps_detected is None:
        _, _, gaps_detected = needs_web_research(product)
    if critical_gaps is None:
        critical_gaps = gaps_detected
 
    logger.info(
        f"Investigador: {product.get('title', '?')} | "
        f"gaps={gaps_detected} | críticos={critical_gaps}"
    )
 
    # ── CACHÉ ────────────────────────────────────────────────────────────────
    fingerprint = build_fingerprint(
        product.get("vendor", ""),
        product.get("productType", ""),
        product.get("title", ""),
    )
 
    cached = await check_enrichment_cache(fingerprint)
    if cached:
        raw_specs = cached.get("technical_specs", {})
        found_cache: dict[str, DataPoint] = {}
 
        for key, val in raw_specs.items():
            if isinstance(val, dict):
                try:
                    found_cache[key] = DataPoint(**val)
                except Exception:
                    found_cache[key] = DataPoint(
                        value       = str(val.get("value", "")),
                        confidence  = ConfidenceLevel.MEDIUM,
                        source_url  = cached.get("source_url", ""),
                        source_type = "cache",
                    )
 
        return ResearchResult(
            found_data        = found_cache,
            not_found         = [],
            source_url        = cached.get("source_url", ""),
            raw_content       = cached.get("raw_content", ""),
            search_query_used = "[desde caché]",
            total_cost_usd    = 0.0,
            rounds_used       = 0,
            from_cache        = True,
        )
 
    # ── BUCLE DE RONDAS ───────────────────────────────────────────────────────
    all_found:    dict[str, DataPoint] = {}
    remaining:    list[str]            = list(gaps_detected)
    all_attempts: list[SearchAttempt]  = []
    total_cost:   float                = 0.0
    round_num:    int                  = 0
 
    conf_priority = {
        ConfidenceLevel.HIGH:   0,
        ConfidenceLevel.MEDIUM: 1,
        ConfidenceLevel.LOW:    2,
    }
 
    while remaining:
        round_num += 1
        logger.info(f"=== RONDA {round_num} === gaps: {remaining}")
 
        # Generar queries
        queries     = await expand_queries(product, remaining, research_instructions)
        total_cost += 0.0001
 
        if not queries:
            logger.warning("Sin queries — abortando ronda")
            break
 
        # Buscar en paralelo
        attempts    = await search_all_queries_parallel(queries, max_parallel=3)
        total_cost += 0.008 * len(queries)
 
        # Costo de Firecrawl por páginas raspadas
        scraped     = sum(
            1 for a in attempts
            if a.raw_content and a.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)
        )
        total_cost += 0.005 * scraped
        all_attempts.extend(attempts)
 
        # Sintetizar
        round_found, round_not_found = await synthesize_multi_source(
            attempts, remaining, product
        )
        total_cost += 0.0003
 
        # Acumular — mayor confianza siempre gana
        for key, dp in round_found.items():
            existing = all_found.get(key)
            if existing is None or conf_priority[dp.confidence] < conf_priority[existing.confidence]:
                all_found[key] = dp
 
        remaining = round_not_found
 
        keep, reason = should_search_more(
            all_found, remaining, round_num, critical_gaps, max_rounds
        )
        logger.info(
            f"Ronda {round_num}: found={len(round_found)}, "
            f"pending={len(remaining)}, continuar={keep} ({reason})"
        )
 
        if not keep:
            break
 
    # ── GUARDAR EN CACHÉ ──────────────────────────────────────────────────────
    best_url    = all_attempts[0].best_url    if all_attempts else ""
    raw_content = all_attempts[0].raw_content if all_attempts else ""
 
    if all_found:
        await save_to_enrichment_cache(
            fingerprint     = fingerprint,
            technical_specs = {k: v.model_dump() for k, v in all_found.items()},
            raw_content     = raw_content,
            source_url      = best_url,
            search_cost_usd = total_cost,
        )
 
    logger.info(
        f"Investigador completado | found={len(all_found)} | "
        f"not_found={len(remaining)} | rondas={round_num} | "
        f"costo=${total_cost:.4f}"
    )
 
    return ResearchResult(
        found_data        = all_found,
        not_found         = remaining,
        source_url        = best_url,
        raw_content       = raw_content,
        search_query_used = f"{round_num} rondas, {len(all_attempts)} queries",
        total_cost_usd    = total_cost,
        rounds_used       = round_num,
        from_cache        = False,
    )
 
 
# ─── NODO LANGGRAPH ──────────────────────────────────────────────────────────
 
async def researcher_node(state: dict) -> dict:
    """
    Nodo de LangGraph.
    Solo se activa si orchestrator_plan.activate_researcher_agent = True.
    Nunca rompe el grafo — si falla, devuelve research_result=None.
    """
    plan = state.get("orchestrator_plan")
 
    if not plan or not getattr(plan, "activate_researcher_agent", False):
        logger.info("Investigador no activado — skip")
        return {**state, "research_result": None}
 
    try:
        result = await research_product(
            product               = state.get("product", {}),
            research_instructions = getattr(plan, "research_instructions", "") or "",
            gaps_detected         = state.get("data_gaps") or None,
            critical_gaps         = state.get("data_gaps") or None,
        )
        return {**state, "research_result": result}
 
    except Exception as e:
        logger.error(f"researcher_node error inesperado: {e}")
        return {**state, "research_result": None}
 
 
# ─── FORMATEADOR PARA EL REDACTOR ────────────────────────────────────────────
 
def format_dossier_for_prompt(result: Optional[ResearchResult]) -> str:
    """
    Convierte ResearchResult en texto estructurado
    listo para inyectar en el prompt del Agente Redactor.
    """
    if not result or not result.found_data:
        return ""
 
    high   = {k: v for k, v in result.found_data.items() if v.confidence == ConfidenceLevel.HIGH}
    medium = {k: v for k, v in result.found_data.items() if v.confidence == ConfidenceLevel.MEDIUM}
    low    = {k: v for k, v in result.found_data.items() if v.confidence == ConfidenceLevel.LOW}
 
    lines = ["## ESPECIFICACIONES TÉCNICAS (INVESTIGACIÓN WEB)\n"]
 
    if high:
        lines.append("### ✅ Verificados (usar con total confianza):")
        for k, dp in high.items():
            lines.append(f"- {k}: {dp.value}")
        lines.append("")
 
    if medium:
        lines.append("### ⚠️ De distribuidor (confiables):")
        for k, dp in medium.items():
            alt = f" [alternativo: {dp.alternative_value}]" if dp.alternative_value else ""
            lines.append(f"- {k}: {dp.value}{alt}")
        lines.append("")
 
    if low:
        lines.append("### 🔴 Fuente secundaria (usar con precaución):")
        for k, dp in low.items():
            if dp.contradiction_detected:
                lines.append(
                    f"- {k}: {dp.value} "
                    f"[CONTRADICCIÓN — otra fuente: {dp.alternative_value}]"
                )
            else:
                lines.append(f"- {k}: {dp.value}")
        lines.append("")
 
    if result.not_found:
        lines.append("### ❌ No encontrado — NO mencionar ni inventar:")
        lines.append(f"- {', '.join(result.not_found)}")
        lines.append("")
 
    footer = (
        f"Fuente: {result.source_url} | "
        f"Costo: ${result.total_cost_usd:.4f} | "
        f"{'caché' if result.from_cache else f'{result.rounds_used} rondas'}"
    )
    lines.append(footer)
 
    return "\n".join(lines)
 