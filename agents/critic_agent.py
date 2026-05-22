from typing import Any

import httpx
from google.genai import errors as genai_errors
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from core.schemas import CriticFeedback
from dotenv import load_dotenv

load_dotenv()

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_RETRIES = 3
STRICT_JSON_OUTPUT_RULE = (
    " STRICT RULE: Your output MUST be a pure JSON object matching the schema. "
    "DO NOT include markdown code blocks like ```json. "
    "DO NOT exceed character limits for titles. "
    "Ensure ALL metadata fields are present."
)


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


primary_critic = Agent(
    model='google-gla:gemini-3.5-flash', 
    output_type=CriticFeedback,
    retries=MAX_RETRIES,
    output_retries=MAX_RETRIES,
    system_prompt=(
        "You are a $100B SaaS Quality Auditor. "
        "Your logic is binary: PERFECT or FAILED. "
        "You have zero tolerance for forbidden words or poor SEO. "
        "Your instructions to the writer must be surgical and direct."
        + STRICT_JSON_OUTPUT_RULE
    )
)

fallback_critic = Agent(
    model='groq:openai/gpt-oss-120b',
    output_type=CriticFeedback,
    retries=MAX_RETRIES,
    output_retries=MAX_RETRIES,
    system_prompt=(
        "You are a strict QA checker for Shopify product copy. "
        "Return FAILED unless all brand rules, forbidden-word rules, and SEO constraints pass. "
        "List concrete issues and short fix instructions. "
        "Return only valid structured data matching the schema."
        + STRICT_JSON_OUTPUT_RULE
    )
)

# Backward-compatible alias for older imports.
critic_agent = primary_critic


async def run_critic_with_fallback(prompt: str) -> Any:
    try:
        return await primary_critic.run(prompt)
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
        return await fallback_critic.run(prompt)
