from typing import Any

import httpx
from google.genai import errors as genai_errors
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from core.schemas import AIProposalOutput
from dotenv import load_dotenv

load_dotenv()

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def _status_code(error: BaseException) -> int | None:
    raw_status = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        return int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        return None


def _is_retryable_provider_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(error, ModelHTTPError):
        status_code = _status_code(error)
        return status_code in RETRYABLE_STATUS_CODES

    if isinstance(error, ModelAPIError):
        status_code = _status_code(error)
        return status_code in RETRYABLE_STATUS_CODES

    if isinstance(error, genai_errors.ServerError):
        return True

    if isinstance(error, (genai_errors.APIError, genai_errors.ClientError)):
        status_code = _status_code(error)
        return status_code in RETRYABLE_STATUS_CODES

    return False


primary_optimizer = Agent(
    model='google-gla:gemini-3.1-pro-preview',
    output_type=AIProposalOutput,
    system_prompt=(
        "You are a $100M/year E-commerce Conversion Rate Optimization (CRO) expert. "
        "Your mission is to rewrite Shopify product listings to maximize sales revenue. "
        "You must STRICTLY adhere to the brand's DNA, formatting rules, and NEVER use forbidden words. "
        "You do not write fluff. You write high-converting, technical, and benefit-driven copy. "
        "Output ONLY valid structured data matching the requested schema."
    )
)

fallback_optimizer = Agent(
    model='groq:llama-3.3-70b-versatile',
    output_type=AIProposalOutput,
    system_prompt=(
        "You are a strict Shopify product copy optimizer. "
        "Rewrite the listing with clear benefits, compliant SEO, and concise conversion copy. "
        "Follow every brand rule exactly. Never use forbidden words. "
        "Keep the title under 70 characters. "
        "Return only valid structured data matching the schema."
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
            "⚠️ [Fallback] Error en Google API (Quota/Timeout). "
            "Activando motor Groq Llama 3..."
        )
        return await fallback_optimizer.run(prompt)
