"""
core/helpers.py
Funciones auxiliares deterministas — sin LLM, sin tokens.
"""

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Timestamp ISO-8601 con zona, apto para columnas timestamptz via PostgREST.

    IMPORTANTE: nunca uses el string "now()" en un dict de update de supabase-py.
    PostgREST lo entrega como literal de texto y Postgres no lo evalúa como función.
    """
    return datetime.now(timezone.utc).isoformat()

# Ruta absoluta a la carpeta skills/ (relativa a este archivo)
SKILLS_DIR = Path(__file__).parent.parent / "skills"


def load_skills(skill_names: list[str]) -> str:
    """
    Lee archivos .json de skills/ y concatena las instrucciones.
    Si un archivo no existe, lo omite con warning (no crash).
    Retorna string vacío si ninguna skill fue encontrada.
    """
    instructions = []

    for name in skill_names:
        skill_path = SKILLS_DIR / f"{name}.json"
        if not skill_path.exists():
            logger.warning(f"Skill no encontrada: {name}.json — omitida")
            continue
        try:
            with open(skill_path, encoding="utf-8") as f:
                skill = json.load(f)
            instruction = skill.get("instruction", "")
            if instruction:
                instructions.append(f"[{skill.get('name', name)}]\n{instruction}")
        except Exception as e:
            logger.error(f"Error leyendo skill {name}: {e}")
            continue

    return "\n\n---\n\n".join(instructions)


def get_available_skills() -> list[str]:
    """
    Retorna lista de nombres de skills disponibles en skills/
    (sin extensión .json).
    """
    if not SKILLS_DIR.exists():
        return []
    return [f.stem for f in SKILLS_DIR.glob("*.json")]


# Categorias que NO son productos fisicos con especificaciones. Un copy
# "tecnico" para una tarjeta de regalo o una descarga digital es un sinsentido.
NON_PHYSICAL_TYPES = {
    "giftcard", "gift card", "gift-card", "digital", "download",
    "service", "servicio", "suscripcion", "subscription",
}


def classify_product_type(product: dict) -> str:
    """
    Clasifica el producto en MANUFACTURED, ARTISANAL, NON_PHYSICAL o GENERIC.
    Usa datos deterministas de Shopify — sin LLM.

    Precedencia (de mayor a menor señal):
        barcode > SKU tecnico > NON_PHYSICAL > ARTISANAL > GENERIC

    barcode y SKU tecnico son identidad del producto, no categoria, por eso
    mandan sobre NON_PHYSICAL. La categoria (product_type) solo decide cuando
    no hay identidad de fabrica.

    MANUFACTURED: tiene barcode O sku con patrón alfanumérico técnico
    NON_PHYSICAL: product_type es una categoria sin especificaciones (tarjeta
                  de regalo, digital, servicio). Compara en minusculas y
                  tolera espacios y guiones.
    ARTISANAL:    sin barcode Y sin sku técnico Y señales de artesanal
    GENERIC:      todo lo demás
    """
    barcode      = (product.get("barcode")      or "").strip()
    sku          = (product.get("sku")          or "").strip()
    product_type = (product.get("product_type") or product.get("productType") or "").lower()

    raw_tags = product.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
    tags = [t.lower() for t in raw_tags]

    # Señales de producto manufacturado (más fuerte: barcode)
    if barcode:
        return "MANUFACTURED"

    # SKU con patrón alfanumérico técnico: "IPH-15-PRO", "3RV1021", "XR500-B"
    TECHNICAL_SKU_PATTERN = re.compile(r'[A-Z]{2,}\d{3,}|[A-Z]{2,}-\d{2,}', re.IGNORECASE)
    if sku and TECHNICAL_SKU_PATTERN.search(sku):
        return "MANUFACTURED"

    # Categoría sin especificaciones físicas. Normaliza guiones a espacios
    # para que "gift-card" y "gift card" caigan en la misma entrada.
    normalized_type = product_type.replace("-", " ").strip()
    if normalized_type in NON_PHYSICAL_TYPES:
        return "NON_PHYSICAL"

    # Señales de producto artesanal
    ARTISANAL_KEYWORDS = {
        "handmade", "hand-made", "artesanal", "artesanía", "hecho a mano",
        "custom", "personalizado", "única", "único", "tejido", "bordado",
        "artisan", "crafted", "handcrafted", "homemade"
    }
    type_is_artisanal = any(kw in product_type for kw in ARTISANAL_KEYWORDS)
    tags_are_artisanal = any(kw in tag for kw in ARTISANAL_KEYWORDS for tag in tags)
    no_codes = not barcode and not sku

    if no_codes and (type_is_artisanal or tags_are_artisanal):
        return "ARTISANAL"

    return "GENERIC"


def calculate_precio_relativo(price: float, avg_price_in_category: float) -> float:
    """
    precio_relativo = precio_producto / avg_precio_nicho
    > 1.5 = premium (necesita justificación de precio)
    ~ 1.0 = precio normal
    < 0.7 = precio bajo (usar urgencia/volumen)
    Maneja división por cero retornando 1.0.
    """
    if not avg_price_in_category or avg_price_in_category <= 0:
        return 1.0
    return round(price / avg_price_in_category, 2)


def map_seo_score_to_category(seo_score: int) -> str:
    """
    Mapea seo_score_initial (0-100) a categoría semántica.
    POOR < 40, ACCEPTABLE 40-69, GOOD 70-89, EXCELLENT >= 90
    """
    if seo_score < 40:
        return "POOR"
    elif seo_score < 70:
        return "ACCEPTABLE"
    elif seo_score < 90:
        return "GOOD"
    return "EXCELLENT"


def build_product_fingerprint(vendor: str, product_type: str, title: str) -> str:
    """
    SHA-256 del modelo base eliminando variantes (color, talla).
    Reutilizado desde researcher_agent para consistencia de caché.
    """
    VARIANT_PATTERNS = [
        r'\b(negro|blanco|rojo|azul|verde|gris|beige|dorado|plateado|rosa|morado)\b',
        r'\b(black|white|red|blue|green|gray|grey|gold|silver|pink|purple|orange)\b',
        r'\b(xs|sm|md|lg|xl|xxl|xxxl|talla|size)\b',
        r'\b(3[6-9]|4[0-9]|5[0-2])\b',
        r'\b(nuevo|new|edicion|edition|especial|special|pack|kit|set|bundle)\b',
    ]

    base = title.lower().strip()
    for pattern in VARIANT_PATTERNS:
        base = re.sub(pattern, '', base, flags=re.IGNORECASE)
    base = ' '.join(base.split())

    raw = f"{vendor.lower().strip()}_{product_type.lower().strip()}_{base}"
    return hashlib.sha256(raw.encode()).hexdigest()


# Tipos de metafield que producen hechos afirmables. Lista blanca por TIPO, no
# por clave: los tipos estructurados de Shopify traen valor Y unidad, asi que
# son verificables. Los de texto libre no: en la tienda real,
# test_data.binding_mount = "Optimistic" es basura de datos generados, y
# pasarsela al escritor produce sinsentidos (verificado 2026-08-18, 3 productos).
VERIFIED_METAFIELD_TYPES = {
    "dimension", "weight", "volume", "rating",
    "number_integer", "number_decimal",
}

_UNIT_LABELS = {
    "CENTIMETERS": "cm", "MILLIMETERS": "mm", "METERS": "m",
    "INCHES": "in", "FEET": "ft",
    "KILOGRAMS": "kg", "GRAMS": "g", "POUNDS": "lb", "OUNCES": "oz",
    "LITERS": "L", "MILLILITERS": "ml",
}


def extract_verified_facts(metafields: dict | None) -> list[str]:
    """Hechos NIVEL 1: emitidos por Shopify, verificados por definicion.

    Son los unicos datos tecnicos que el Juez debe aceptar sin discusion. El
    bucle de invencion del producto 1010 (2026-08-17) ocurrio teniendo este
    metafield disponible y descartado: el Juez veto "nucleo de madera" por no
    verificable, pero habria aceptado "Longitud: 159 cm" sin objecion.

    Determinista, sin LLM y sin red.
    """
    if not metafields:
        return []

    facts: list[str] = []
    for full_key, entry in metafields.items():
        if not isinstance(entry, dict):
            continue
        mtype = (entry.get("type") or "").strip()
        if mtype not in VERIFIED_METAFIELD_TYPES:
            continue

        raw = entry.get("value")
        label = full_key.split(".")[-1].replace("_", " ").strip().capitalize()

        parsed = raw
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = raw

        if isinstance(parsed, dict) and "value" in parsed:
            value = parsed.get("value")
            unit = _UNIT_LABELS.get(str(parsed.get("unit") or "").upper(), "")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            facts.append(f"{label}: {value} {unit}".strip())
        elif parsed not in (None, ""):
            facts.append(f"{label}: {parsed}")

    return facts
