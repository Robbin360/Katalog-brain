from typing import Any

import httpx


SHOPIFY_API_VERSION = "2026-04"


# ─── Excepciones tipadas (publicación) ──────────────────────────────────────
# El consumidor (core/publish_recovery.py) clasifica reintentables vs
# permanentes por TIPO, no por sniffing de strings. Los fallos de red/timeout
# que hoy eran RuntimeError ambiguos ahora tienen clase propia.


class ShopifyError(Exception):
    """Error base de la integración Shopify Admin GraphQL (publicación)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ShopifyTimeoutError(ShopifyError):
    pass


class ShopifyNetworkError(ShopifyError):
    pass


class ShopifyRateLimitError(ShopifyError):
    pass


class ShopifyAuthError(ShopifyError):
    pass


class ShopifyNotFoundError(ShopifyError):
    pass


class ShopifyServerError(ShopifyError):
    pass


class ShopifyGraphQLError(ShopifyError):
    def __init__(self, message: str, graphql_errors: list | None = None) -> None:
        super().__init__(message)
        self.graphql_errors = graphql_errors or []


class ShopifyValidationError(ShopifyError):
    def __init__(self, message: str, user_errors: list | None = None) -> None:
        super().__init__(message)
        self.user_errors = user_errors or []


class ShopifyConsistencyError(ShopifyError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


def _normalize_shop_url(shop_url: str) -> str:
    clean_url = shop_url.strip()
    clean_url = clean_url.removeprefix("https://").removeprefix("http://")
    return clean_url.rstrip("/")


def _normalize_product_gid(product_shopify_id: str) -> str:
    clean_id = str(product_shopify_id).strip()
    if clean_id.startswith("gid://shopify/Product/"):
        return clean_id
    return f"gid://shopify/Product/{clean_id}"


def _raise_for_http_status(response: httpx.Response) -> None:
    """Traduce códigos HTTP a excepciones tipadas (nunca httpx crudo)."""
    status = response.status_code
    if status in (401, 403):
        raise ShopifyAuthError(
            f"Shopify authentication failed (HTTP {status}).",
            status_code=status,
        )
    if status == 404:
        raise ShopifyNotFoundError(
            f"Shopify resource not found (HTTP {status}).",
            status_code=status,
        )
    if status == 429:
        raise ShopifyRateLimitError(
            f"Shopify rate limit exceeded (HTTP {status}).",
            status_code=status,
        )
    if status >= 500:
        raise ShopifyServerError(
            f"Shopify server error (HTTP {status}).",
            status_code=status,
        )
    response.raise_for_status()


async def _post_graphql_json(
    endpoint: str,
    headers: dict[str, str],
    query: str,
    variables: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """POST GraphQL con errores de transporte tipados. Nunca lanza httpx."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                endpoint,
                json={"query": query, "variables": variables or {}},
                headers=headers,
            )
        except httpx.TimeoutException as e:
            raise ShopifyTimeoutError(f"Shopify request timed out: {e}") from e
        except httpx.NetworkError as e:
            raise ShopifyNetworkError(f"Shopify network error: {e}") from e

    _raise_for_http_status(response)

    try:
        return response.json()
    except ValueError as e:
        raise ShopifyError(
            f"Shopify returned a non-JSON response (HTTP {response.status_code})."
        ) from e


async def get_product_copy(
    shop_url: str,
    access_token: str,
    product_shopify_id: str,
) -> dict[str, Any]:
    """Lee el título y la descripción HTML actuales de un producto Shopify.

    Sirve a la recuperación idempotente de publicación: si el contenido ya
    coincide con la propuesta aprobada, el Nodo 6 no reescribe.

    Retorna {"title": str, "descriptionHtml": str}.
    Lanza subclases de ShopifyError ante cualquier fallo (nunca httpx).
    """
    if not shop_url:
        raise ValueError("Missing Shopify shop_url")
    if not access_token:
        raise ValueError("Missing Shopify access_token")
    if not product_shopify_id:
        raise ValueError("Missing Shopify product ID")

    normalized_shop_url = _normalize_shop_url(shop_url)
    product_gid = _normalize_product_gid(product_shopify_id)
    endpoint = (
        f"https://{normalized_shop_url}/admin/api/"
        f"{SHOPIFY_API_VERSION}/graphql.json"
    )

    query = """
    query GetProductCopy($id: ID!) {
      product(id: $id) {
        id
        title
        descriptionHtml
      }
    }
    """

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }

    response_data = await _post_graphql_json(
        endpoint, headers, query, {"id": product_gid}
    )

    graph_errors = response_data.get("errors") or []
    if graph_errors:
        message = graph_errors[0].get("message", str(graph_errors))
        raise ShopifyGraphQLError(
            f"Shopify GraphQL errors: {message}",
            graph_errors=graph_errors,
        )

    product = (response_data.get("data") or {}).get("product")
    if not product:
        raise ShopifyNotFoundError(
            f"Shopify product {product_shopify_id} not found"
        )

    return {
        "title": product.get("title", ""),
        "descriptionHtml": product.get("descriptionHtml", ""),
    }


async def publish_to_shopify(
    shop_url: str,
    access_token: str,
    product_shopify_id: str,
    title: str,
    html: str,
) -> dict[str, Any]:
    """
    Update Shopify product copy through Admin GraphQL.
    Uses ProductUpdateInput.descriptionHtml for the product HTML description.
    Lanza subclases de ShopifyError ante cualquier fallo (nunca httpx).
    """
    if not shop_url:
        raise ValueError("Missing Shopify shop_url")
    if not access_token:
        raise ValueError("Missing Shopify access_token")
    if not product_shopify_id:
        raise ValueError("Missing Shopify product ID")

    normalized_shop_url = _normalize_shop_url(shop_url)
    product_gid = _normalize_product_gid(product_shopify_id)
    endpoint = (
        f"https://{normalized_shop_url}/admin/api/"
        f"{SHOPIFY_API_VERSION}/graphql.json"
    )

    mutation = """
    mutation UpdateKatalogProduct($product: ProductUpdateInput!) {
      productUpdate(product: $product) {
        product {
          id
          title
        }
        userErrors {
          field
          message
        }
      }
    }
    """

    payload = {
        "query": mutation,
        "variables": {
            "product": {
                "id": product_gid,
                "title": title,
                "descriptionHtml": html,
            }
        },
    }

    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }

    response_data = await _post_graphql_json(
        endpoint, headers, mutation, payload["variables"]
    )

    graph_errors = response_data.get("errors") or []
    if graph_errors:
        message = graph_errors[0].get("message", str(graph_errors))
        raise ShopifyGraphQLError(
            f"Shopify GraphQL errors: {message}",
            graph_errors=graph_errors,
        )

    update_payload = (response_data.get("data") or {}).get("productUpdate") or {}
    user_errors = update_payload.get("userErrors") or []
    if user_errors:
        messages = [ue.get("message", str(ue)) for ue in user_errors]
        raise ShopifyValidationError(
            f"Shopify user errors: {'; '.join(messages)}",
            user_errors=user_errors,
        )

    product = update_payload.get("product")
    if not product:
        raise ShopifyError("Shopify did not return an updated product")

    returned_title = (product or {}).get("title", "")
    if returned_title != title:
        print(f"🔴 [Shopify] Respuesta completa: {response_data}")
        raise ShopifyConsistencyError(
            f"Shopify aceptó la mutación pero el título no cambió. "
            f"Enviado: {title!r}. Devuelto: {returned_title!r}.",
            details={"sent_title": title, "returned_title": returned_title},
        )

    return product
