from typing import Any

import httpx
from google.genai import errors as genai_errors
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from core.model_config import OPTIMIZER_FALLBACK_MODEL, OPTIMIZER_PRIMARY_MODEL
from core.provider_errors import is_retryable_provider_error
from core.schemas import AIProposalOutput

MAX_RETRIES = 3
STRICT_JSON_OUTPUT_RULE = (
    " STRICT RULE: Your output MUST be a pure JSON object matching the schema. "
    "DO NOT include markdown code blocks like ```json. "
    "DO NOT exceed character limits for titles. "
    "Ensure ALL metadata fields are present."
)

_is_retryable_provider_error = is_retryable_provider_error  # alias local


primary_optimizer = Agent(
    model=OPTIMIZER_PRIMARY_MODEL,
    output_type=AIProposalOutput,
    retries=MAX_RETRIES,
    output_retries=MAX_RETRIES,
    defer_model_check=True,
    system_prompt=(
        "You are a $100M/year E-commerce Conversion Rate Optimization (CRO) expert. "
        "Your mission is to rewrite Shopify product listings to maximize sales revenue. "
        "You must STRICTLY adhere to the brand's DNA, formatting rules, and NEVER use forbidden words. "
        "You do not write fluff. You write high-converting, technical, and benefit-driven copy. "
        "The title MUST be between 40 and 70 characters: never shorter than 40. "
        "Output ONLY valid structured data matching the requested schema."
        + STRICT_JSON_OUTPUT_RULE
    )
)

fallback_optimizer = Agent(
    model=OPTIMIZER_FALLBACK_MODEL,
    output_type=AIProposalOutput,
    retries=MAX_RETRIES,
    output_retries=MAX_RETRIES,
    defer_model_check=True,
    system_prompt=(
        "You are a strict Shopify product copy optimizer. "
        "Rewrite the listing with clear benefits, compliant SEO, and concise conversion copy. "
        "Follow every brand rule exactly. Never use forbidden words. "
        "The title MUST be between 40 and 70 characters: never shorter than 40. "
        "Return only valid structured data matching the schema."
        + STRICT_JSON_OUTPUT_RULE
    )
)

# Backward-compatible alias for older imports.
optimizer_agent = primary_optimizer


async def run_optimizer_with_fallback(prompt: str) -> Any:
    try:
        return await primary_optimizer.run(prompt)
    except (
        ModelHTTPError,
        ModelAPIError,
        genai_errors.APIError,
        httpx.HTTPError,
        TimeoutError,
    ) as e:
        if not _is_retryable_provider_error(e):
            raise

        print(
            f"⚠️ [Fallback] Proveedor primario falló "
            f"({type(e).__name__}: {e}). Activando modelo de respaldo..."
        )
        return await fallback_optimizer.run(prompt)
    except Exception as e:
        error_msg = str(e).lower()
        if "output validation" in error_msg or "retries" in error_msg:
            print(
                f"⚠️ [Fallback] Error de validación en el primario "
                f"({type(e).__name__}: {e}). Activando modelo de respaldo..."
            )
            return await fallback_optimizer.run(prompt)
        raise
