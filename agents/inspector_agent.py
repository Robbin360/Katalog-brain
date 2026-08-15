"""
agents/inspector_agent.py

Juez de RELEVANCIA y CORRECCIÓN. No puntúa: veta.

Por qué cambió (error analysis del 2026-08-15 sobre 18 productos reales):

1. El inspector anterior daba 0-100 sobre "emotional hook, SEO clarity,
   readability". Medido contra el catálogo, esos puntajes estaban
   invertidos: el 1011 ("Tabla Pro 3p") sacó 95 y el 1009, el mejor
   texto del catálogo, sacó 25. Una escala sin calibrar entre tres
   dimensiones abstractas no distingue calidad.

2. Aprobó "maximizar el avión en superficies gélidas" (traducción rota
   de "glide" como "avión") porque el texto sí tenía gancho, keywords y
   formato. Ninguno de sus criterios miraba si el español era correcto.

3. No pudo detectar que el producto 1004 (snowboard) recibió una
   propuesta titulada "Estabilizador de Video Profesional para Cámara y
   Cine": nunca supo de qué producto se trataba.

Diseño nuevo, siguiendo la práctica de evals específicos y binarios en
lugar de métricas genéricas:
  - dos veredictos booleanos, no una escala;
  - cada fallo exige CITAR el fragmento culpable, lo que obliga al
    modelo a fundamentar y hace el veredicto auditable;
  - recibe una REFERENCIA de identidad del producto porque la relevancia
    no existe sin punto de comparación;
  - el prompt está en español porque los productos están en español: un
    juez que opera en inglés detecta peor una traducción rota al español.

Lo que este agente NO hace, a propósito:
  - No valida especificaciones inventadas. Eso requiere el dossier de
    product_enrichment, que no recibe; es trabajo del critic.
  - No mide forma (longitud, estructura, stuffing). Eso lo hace
    core/deterministic_score.py sin gastar tokens.
"""

from __future__ import annotations

import asyncio
import random
import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from core.model_config import INSPECTOR_MODEL


class RelevanceVerdict(BaseModel):
    """Veredicto binario. Sin escalas: pasa o no pasa."""

    same_product: bool = Field(
        ...,
        description=(
            "True si el texto describe el MISMO producto que la referencia. "
            "False si describe otro producto o una categoría distinta."
        ),
    )
    language_ok: bool = Field(
        ...,
        description=(
            "True si el español es correcto y comprensible. "
            "False si hay traducción literal, concordancia rota o frases "
            "sin sentido."
        ),
    )
    evidence: str = Field(
        ...,
        description=(
            "Si algún veredicto es False, CITA TEXTUAL del fragmento culpable "
            "y explicación en una frase. Si ambos son True, escribe 'OK'."
        ),
    )

    @property
    def passed(self) -> bool:
        return self.same_product and self.language_ok

    def as_log_line(self) -> str:
        if self.passed:
            return "Inspector APROBADO: producto correcto, lenguaje correcto."
        problemas = []
        if not self.same_product:
            problemas.append("producto equivocado")
        if not self.language_ok:
            problemas.append("lenguaje incorrecto")
        return f"Inspector RECHAZADO ({', '.join(problemas)}): {self.evidence}"


INSPECTOR_SETTINGS = {"temperature": 0}

inspector_agent = Agent(
    model=INSPECTOR_MODEL,
    output_type=RelevanceVerdict,
    defer_model_check=True,
    model_settings=INSPECTOR_SETTINGS,
    system_prompt=(
        "Eres un revisor editorial de fichas de producto de e-commerce en "
        "español. Tu único trabajo es detectar dos fallos graves. No evalúas "
        "persuasión, ni SEO, ni formato: eso lo mide otro sistema.\n"
        "\n"
        "FALLO 1 — PRODUCTO EQUIVOCADO (same_product = false)\n"
        "El texto describe un producto distinto al de la referencia.\n"
        "Ejemplo real de fallo: la referencia es una tabla de snowboard y el "
        "texto propuesto se titula 'Estabilizador de Video Profesional para "
        "Cámara y Cine'. Son productos diferentes: same_product = false.\n"
        "NO es fallo: reformular, cambiar el enfoque, usar sinónimos, "
        "añadir la categoría o quitar la marca. Mientras sea el mismo "
        "objeto físico, same_product = true.\n"
        "\n"
        "FALLO 2 — LENGUAJE INCORRECTO (language_ok = false)\n"
        "El español está roto: traducción literal del inglés, concordancia "
        "incorrecta, o frases sin sentido.\n"
        "Ejemplo real de fallo: 'Base de baja fricción diseñada para "
        "maximizar el avión en superficies gélidas'. Es 'glide' traducido "
        "como 'avión' en vez de 'planeo': no tiene sentido, "
        "language_ok = false.\n"
        "Otro ejemplo real: 'se adapta a cualquier terreno elevado y releva "
        "técnicos' no es español gramatical: language_ok = false.\n"
        "NO es fallo: vocabulario técnico, frases largas, anglicismos "
        "aceptados del sector (rider, freestyle, all-mountain, snowboard).\n"
        "\n"
        "REGLAS DE SALIDA\n"
        "Si marcas algo como false, en 'evidence' debes citar textualmente "
        "el fragmento culpable entre comillas y explicar el problema en una "
        "frase. Si no puedes citar un fragmento concreto, entonces NO hay "
        "fallo y debes marcar true.\n"
        "Si ambos veredictos son true, 'evidence' debe ser exactamente 'OK'.\n"
        "Ante la duda, aprueba: un falso rechazo bloquea trabajo bueno."
    ),
)


def build_inspector_prompt(
    reference_title: str | None,
    reference_vendor: str | None,
    reference_tags: str | None,
    candidate_title: str | None,
    candidate_body_html: str | None,
) -> str:
    """
    Serialización única del caso para el inspector.

    La REFERENCIA es la identidad del producto según la tienda. No existe
    columna product_type en shopify_products (verificado por introspección
    el 2026-08-15), así que usamos las señales que sí están: título actual,
    vendor y tags.
    """
    return (
        "REFERENCIA (identidad del producto segun la tienda)\n"
        f"Titulo actual: {reference_title or '(sin titulo)'}\n"
        f"Marca: {reference_vendor or '(sin marca)'}\n"
        f"Etiquetas: {reference_tags or '(sin etiquetas)'}\n"
        "\n"
        "TEXTO A REVISAR\n"
        f"Titulo propuesto: {candidate_title or ''}\n"
        f"Descripcion propuesta: {candidate_body_html or ''}"
    )


# ── REINTENTO ANTE RATE LIMIT ────────────────────────────────────────
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


async def check_relevance(
    *,
    reference_title: str | None,
    reference_vendor: str | None = None,
    reference_tags: str | None = None,
    candidate_title: str | None,
    candidate_body_html: str | None,
    max_attempts: int = 4,
) -> RelevanceVerdict:
    """
    Un solo llamado al modelo. A diferencia del gate anterior, que puntuaba
    el texto viejo Y el nuevo (2 requests), relevancia y corrección son
    propiedades absolutas del candidato: basta evaluarlo una vez.
    """
    prompt = build_inspector_prompt(
        reference_title,
        reference_vendor,
        reference_tags,
        candidate_title,
        candidate_body_html,
    )
    backoff = 2.0

    for attempt in range(1, max_attempts + 1):
        try:
            result = await inspector_agent.run(prompt)
            return result.output
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == max_attempts:
                raise
            match = _RETRY_DELAY_RE.search(str(exc))
            wait = float(match.group(1)) + 1.0 if match else backoff
            await asyncio.sleep(wait + random.uniform(0, 0.75))
            backoff = min(backoff * 2, 30.0)

    raise RuntimeError("check_relevance: bucle de reintentos terminado sin resultado")