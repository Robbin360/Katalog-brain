"""
core/publish_recovery.py
========================
Recuperación de publicaciones Shopify: clasificación de fallos, backoff con
tope y persistencia estructurada vía RPC `record_publish_failure`.

Diseño (sin tablas ni estados de auditoría nuevos):
  - Fallo REINTENTABLE con intentos disponibles → el producto se queda en
    READY_TO_PUBLISH (la propuesta sigue aprobada y pagada) y se programa
    `publish_next_retry_at` con backoff exponencial con tope. El Auto-Pilot
    lo vuelve a tomar cuando vence la ventana (filtro en core/worker.py).
  - Fallo PERMANENTE o intentos agotados → ERROR con retry_attempts=3
    (congelación: el filtro de elegibilidad del Auto-Pilot exige
    retry_attempts < 3), más aviso al sistema de compensación.
"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from core.shopify_tools import (
    ShopifyAuthError,
    ShopifyConsistencyError,
    ShopifyGraphQLError,
    ShopifyNetworkError,
    ShopifyNotFoundError,
    ShopifyRateLimitError,
    ShopifyServerError,
    ShopifyTimeoutError,
    ShopifyValidationError,
)

# Intentos de publicación antes de congelar el producto en ERROR (5 fallos ≈
# 2h40m de ventanas: 5+10+20+40+80 minutos).
MAX_PUBLISH_ATTEMPTS = 5
PUBLISH_RETRY_BASE_MINUTES = 5
PUBLISH_RETRY_CAP_MINUTES = 240

# Etapas donde puede fallar una publicación. Se persisten en
# `publish_error_stage` y en el RPC.
PUBLISH_STAGE_SETUP = "brain_setup"      # integración/token/decrypt (brain-side)
PUBLISH_STAGE_VERIFY = "shopify_verify"  # lectura idempotente del contenido actual
PUBLISH_STAGE_UPDATE = "shopify_update"  # mutación productUpdate
PUBLISH_STAGE_PERSIST = "brain_persist"  # bookkeeping local tras confirmar Shopify

PUBLISH_CODE_RATE_LIMIT = "RATE_LIMIT"
PUBLISH_CODE_TIMEOUT = "TIMEOUT"
PUBLISH_CODE_NETWORK = "NETWORK"
PUBLISH_CODE_SHOPIFY_SERVER = "SHOPIFY_SERVER"
PUBLISH_CODE_AUTH = "AUTH_ERROR"
PUBLISH_CODE_NOT_FOUND = "NOT_FOUND"
PUBLISH_CODE_VALIDATION = "VALIDATION"
PUBLISH_CODE_CONSISTENCY = "CONSISTENCY_CHECK_FAILED"
PUBLISH_CODE_GRAPHQL = "GRAPHQL_ERROR"
PUBLISH_CODE_UNKNOWN = "UNKNOWN"

# Errores GraphQL del Admin API que no se curan solos: credenciales rotas,
# producto borrado, payload inválido. Todo lo demás (internal server, etc.)
# se trata como reintentable.
_NON_RETRYABLE_GRAPHQL_MARKERS = (
    "access denied",
    "permission",
    "unauthorized",
    "forbidden",
    "not found",
    "does not exist",
    "must not",
    "invalid",
    "exceeds",
    "too long",
)

MAX_MESSAGE_LENGTH = 2000
MAX_DETAILS_JSON_LENGTH = 8000


@dataclass
class PublishFailure:
    """Diagnóstico estructurado de un fallo de publicación (contrato del RPC)."""

    code: str
    stage: str
    retryable: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def classify_publish_error(exc: BaseException, stage: str) -> PublishFailure:
    """Traduce una excepción de publicación a un PublishFailure tipado.

    Retryable = el MISMO intento puede triunfar sin intervención humana
    (rate limit, timeout, 5xx, caídas de red). Lo demás congela el producto:
    reintentar un payload rechazado o unas credenciales rotas solo duplica
    fallos y ruido.
    """
    message = _truncate(str(exc) or exc.__class__.__name__, MAX_MESSAGE_LENGTH)

    if isinstance(exc, ShopifyRateLimitError):
        return PublishFailure(PUBLISH_CODE_RATE_LIMIT, stage, True, message, {"http_status": 429})
    if isinstance(exc, ShopifyTimeoutError):
        return PublishFailure(PUBLISH_CODE_TIMEOUT, stage, True, message)
    if isinstance(exc, ShopifyNetworkError):
        return PublishFailure(PUBLISH_CODE_NETWORK, stage, True, message)
    if isinstance(exc, ShopifyServerError):
        return PublishFailure(
            PUBLISH_CODE_SHOPIFY_SERVER, stage, True, message,
            {"http_status": exc.status_code},
        )
    if isinstance(exc, ShopifyAuthError):
        return PublishFailure(
            PUBLISH_CODE_AUTH, stage, False, message,
            {"http_status": exc.status_code},
        )
    if isinstance(exc, ShopifyNotFoundError):
        return PublishFailure(PUBLISH_CODE_NOT_FOUND, stage, False, message)
    if isinstance(exc, ShopifyValidationError):
        return PublishFailure(
            PUBLISH_CODE_VALIDATION, stage, False, message,
            {"user_errors": exc.user_errors},
        )
    if isinstance(exc, ShopifyConsistencyError):
        return PublishFailure(PUBLISH_CODE_CONSISTENCY, stage, True, message, exc.details)
    if isinstance(exc, ShopifyGraphQLError):
        retryable = not any(
            marker in message.lower() for marker in _NON_RETRYABLE_GRAPHQL_MARKERS
        )
        return PublishFailure(
            PUBLISH_CODE_GRAPHQL, stage, retryable, message,
            {"graphql_errors": exc.graphql_errors},
        )

    return PublishFailure(
        PUBLISH_CODE_UNKNOWN, stage, False, message,
        {"exception_type": exc.__class__.__name__},
    )


def publish_next_retry_iso(publish_attempts: int) -> str:
    """Ventana del próximo reintento tras un fallo (backoff exponencial con tope).

    `publish_attempts` es el conteo DESPUÉS del fallo que se acaba de registrar
    (1 → 5 min, 2 → 10 min, 3 → 20 min, 4 → 40 min, 5 → 80 min, tope 4 h).
    """
    attempts = max(int(publish_attempts or 0), 1)
    minutes = PUBLISH_RETRY_BASE_MINUTES * (2 ** (attempts - 1))
    minutes = min(minutes, PUBLISH_RETRY_CAP_MINUTES)
    return (datetime.now(timezone.utc) + timedelta(minutes=minutes)).strftime(
        '%Y-%m-%dT%H:%M:%SZ'
    )


def _bounded_details(details: dict[str, Any]) -> dict[str, Any]:
    """Corta details JSON si excede el límite (jsonb no debería inflarse)."""
    encoded = json.dumps(details, ensure_ascii=False, default=str)
    if len(encoded) <= MAX_DETAILS_JSON_LENGTH:
        return details
    return {
        "truncated": True,
        "summary": encoded[: MAX_DETAILS_JSON_LENGTH - 1] + "…",
    }


async def record_publish_failure(
    user_id: str,
    product_id: int,
    failure: PublishFailure,
    next_retry_at: str | None,
    shopify_confirmed: bool = False,
) -> None:
    """Persiste el fallo vía RPC.

    Import lazy de supabase para evitar el ciclo graph ⇄ publish_recovery.
    El llamador es responsable de la transición de audit_status.
    """
    from core.graph import supabase

    def _call():
        return supabase.rpc("record_publish_failure", {
            "p_user_id": user_id,
            "p_product_id": product_id,
            "p_code": failure.code,
            "p_stage": failure.stage,
            "p_retryable": failure.retryable,
            "p_message": failure.message,
            "p_details": _bounded_details(failure.details),
            "p_next_retry_at": next_retry_at,
            "p_shopify_confirmed": shopify_confirmed,
        }).execute()

    await asyncio.to_thread(_call)
