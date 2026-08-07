from __future__ import annotations

import asyncio
import random
import re

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from core.model_config import INSPECTOR_MODEL


# 1. EL MOLDE ESTRICTO
class InspectorResult(BaseModel):
    score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Puntuación de salud SEO y Ventas (0-100).",
    )
    reason: str = Field(
        ...,
        description="Máximo 15 palabras. Razón brutalmente honesta de por qué el texto vende o fracasa.",
    )


# 2. DETERMINISMO OBLIGATORIO
INSPECTOR_SETTINGS = {"temperature": 0}

inspector_agent = Agent(
    model=INSPECTOR_MODEL,
    output_type=InspectorResult,
    defer_model_check=True,
    model_settings=INSPECTOR_SETTINGS,
    system_prompt=(
        "You are a ruthless E-commerce Conversion Rate (CRO) and SEO Auditor. "
        "I will give you a Shopify product title and its HTML description. "
        "Evaluate it strictly on: "
        "1. Emotional hook (Does it make me want to buy?) "
        "2. SEO clarity (Are there clear keywords?) "
        "3. Readability (Is it a wall of text or well-formatted?). "
        "Return a score from 0 to 100 and a very short, punchy reason for your score."
    ),
)


# 3. FORMATO ÚNICO DEL PROMPT
def build_inspector_prompt(title: str | None, body_html: str | None) -> str:
    """Serialización única del producto para el inspector."""
    return f"Title: {title or ''}\nDescription: {body_html or ''}"


# 4. REINTENTO ANTE RATE LIMIT
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+(?:\.\d+)?)s")


def _is_rate_limit(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


async def score_content(
    title: str | None,
    body_html: str | None,
    *,
    max_attempts: int = 4,
) -> tuple[int, str]:
    """Puntúa un texto. Devuelve (score, reason)."""
    prompt = build_inspector_prompt(title, body_html)
    backoff = 2.0

    for attempt in range(1, max_attempts + 1):
        try:
            result = await inspector_agent.run(prompt)
            return result.output.score, result.output.reason
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == max_attempts:
                raise
            match = _RETRY_DELAY_RE.search(str(exc))
            wait = float(match.group(1)) + 1.0 if match else backoff
            await asyncio.sleep(wait + random.uniform(0, 0.75))
            backoff = min(backoff * 2, 30.0)

    raise RuntimeError("score_content: bucle de reintentos terminado sin resultado")
