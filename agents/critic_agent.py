from typing import Any

import httpx
from google.genai import errors as genai_errors
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from core.model_config import CRITIC_FALLBACK_MODEL, CRITIC_PRIMARY_MODEL
from core.schemas import CriticFeedback

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
    model=CRITIC_PRIMARY_MODEL, 
    output_type=CriticFeedback,
    retries=MAX_RETRIES,
    output_retries=MAX_RETRIES,
    defer_model_check=True,
     model_settings={"temperature": 0},   
    system_prompt=(
        "You are a $100B SaaS Quality Auditor. "
        "Your logic is binary: PERFECT or FAILED. "
        "You have zero tolerance for forbidden words or poor SEO. "
        "Your instructions to the writer must be surgical and direct."
        + STRICT_JSON_OUTPUT_RULE
    )
)

fallback_critic = Agent(
    model=CRITIC_FALLBACK_MODEL,
    output_type=CriticFeedback,
    retries=MAX_RETRIES,
    output_retries=MAX_RETRIES,
    defer_model_check=True,
     model_settings={"temperature": 0},   
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


def _to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def _get_attr_or_key(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def pre_critic_checks(proposal: Any, rules: Any) -> None:
    proposal_dict = _to_dict(proposal)
    title = str(proposal_dict.get("new_title", ""))
    body_html = str(proposal_dict.get("new_body_html", ""))
    seo_tags = str(proposal_dict.get("seo_tags", ""))
    forbidden_words = _get_attr_or_key(rules, "forbidden_words", []) or []
    haystack = f"{title}\n{body_html}\n{seo_tags}".lower()

    failures: list[str] = []
    if len(title) > 70:
        failures.append(f"Title exceeds 70 characters ({len(title)}).")

    blocked_words = [
        str(word)
        for word in forbidden_words
        if str(word).strip() and str(word).strip().lower() in haystack
    ]
    if blocked_words:
        failures.append(f"Forbidden words found: {', '.join(blocked_words)}.")

    if failures:
        raise ValueError("Pre-critic deterministic checks failed: " + " ".join(failures))


async def run_critic_with_fallback(prompt: str, proposal: Any = None, rules: Any = None) -> Any:
    if proposal is not None and rules is not None:
        pre_critic_checks(proposal, rules)

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
            f"⚠️ [Fallback] El proveedor primario del juez falló "
            f"({type(e).__name__}: {e}). Activando modelo de respaldo..."
        )
        return await fallback_critic.run(prompt)
