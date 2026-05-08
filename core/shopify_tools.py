from typing import Any

import httpx


SHOPIFY_API_VERSION = "2026-04"


def _normalize_shop_url(shop_url: str) -> str:
    clean_url = shop_url.strip()
    clean_url = clean_url.removeprefix("https://").removeprefix("http://")
    return clean_url.rstrip("/")


def _normalize_product_gid(product_shopify_id: str) -> str:
    clean_id = str(product_shopify_id).strip()
    if clean_id.startswith("gid://shopify/Product/"):
        return clean_id
    return f"gid://shopify/Product/{clean_id}"


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

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        response_data: dict[str, Any] = response.json()

    graph_errors = response_data.get("errors") or []
    if graph_errors:
        raise RuntimeError(f"Shopify GraphQL errors: {graph_errors}")

    update_payload = (response_data.get("data") or {}).get("productUpdate") or {}
    user_errors = update_payload.get("userErrors") or []
    if user_errors:
        raise RuntimeError(f"Shopify user errors: {user_errors}")

    product = update_payload.get("product")
    if not product:
        raise RuntimeError("Shopify did not return an updated product")

    return product
