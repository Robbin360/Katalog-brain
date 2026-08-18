"""
core/provider_errors.py
=======================
Fuente única para clasificar fallos transitorios de proveedores LLM.

Historia: `RETRYABLE_STATUS_CODES`, `status_code()` e
`is_retryable_provider_error()` estaban duplicados IDÉNTICOS en
agents/critic_agent.py y agents/optimizer_agent.py. Antes de añadir una
tercera copia (el fallback del Orquestador), se extraen aquí con nombres
públicos. El patrón de fallback primario→respaldo de los tres agentes usa
este módulo para decidir si un error (429, 5xx, timeouts, caídas del
proveedor) merece cambiar de modelo o debe propagarse.
"""

import httpx
from google.genai import errors as genai_errors
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError

RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def status_code(error: BaseException) -> int | None:
    raw_status = getattr(error, "status_code", None) or getattr(error, "code", None)
    try:
        return int(raw_status) if raw_status is not None else None
    except (TypeError, ValueError):
        return None


def is_retryable_provider_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, httpx.TimeoutException, httpx.TransportError)):
        return True

    if isinstance(error, ModelHTTPError):
        code = status_code(error)
        return code in RETRYABLE_STATUS_CODES

    if isinstance(error, ModelAPIError):
        code = status_code(error)
        return code in RETRYABLE_STATUS_CODES

    if isinstance(error, genai_errors.ServerError):
        return True

    if isinstance(error, (genai_errors.APIError, genai_errors.ClientError)):
        code = status_code(error)
        return code in RETRYABLE_STATUS_CODES

    return False