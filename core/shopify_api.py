import asyncio
from typing import Any
import httpx

SHOPIFY_API_VERSION = "2026-04"

def _numeric_id_to_gid(numeric_id: str) -> str:
    """
    Convierte un ID numérico o un ID parcial al GID completo de Shopify para un Producto.
    E.g. "12345" -> "gid://shopify/Product/12345"
    Si ya es un GID completo, lo retorna idéntico.
    """
    clean_id = str(numeric_id).strip()
    if clean_id.startswith("gid://shopify/Product/"):
        return clean_id
    if "/" in clean_id:
        clean_id = clean_id.split("/")[-1]
    return f"gid://shopify/Product/{clean_id}"

def _normalize_shop_url(shop_url: str) -> str:
    clean_url = shop_url.strip()
    clean_url = clean_url.removeprefix("https://").removeprefix("http://")
    return clean_url.rstrip("/")

async def get_product_taxonomy(
    shopify_numeric_id: str,
    shop_domain: str,
    access_token: str,
) -> tuple[str, bool]:
    """
    Consulta la categoría y atributos de taxonomía predictiva de Shopify para un producto.
    Retorna (prompt_text, success_flag).
    Implementa retry con exponential backoff para HTTP 429 y Timeout. Max 3 intentos.
    Nunca lanza excepciones, retorna ("", False) si falla.
    """
    if not shopify_numeric_id or not shop_domain or not access_token:
        print("⚠️ [shopify_api] get_product_taxonomy recibió parámetros vacíos.")
        return ("", False)

    product_gid = _numeric_id_to_gid(shopify_numeric_id)
    normalized_shop_domain = _normalize_shop_url(shop_domain)
    endpoint = f"https://{normalized_shop_domain}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"

    # Query 1: Obtener la categoría del producto
    product_query = """
    query GetProductCategory($id: ID!) {
      product(id: $id) {
        productCategory {
          productTaxonomyNode {
            id
            fullName
          }
        }
      }
    }
    """

    headers = {
      "Content-Type": "application/json",
      "X-Shopify-Access-Token": access_token,
    }

    async def post_graphql(query: str, variables: dict[str, Any]) -> dict[str, Any]:
        attempt = 1
        max_attempts = 3
        backoff_base = 2

        while True:
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        endpoint,
                        json={"query": query, "variables": variables},
                        headers=headers
                    )
                    # Si es 429, reintentamos con backoff
                    if response.status_code == 429:
                        if attempt < max_attempts:
                            sleep_time = backoff_base ** attempt
                            print(f"⚠️ [shopify_api] HTTP 429 (Rate Limit). Reintentando en {sleep_time}s (Intento {attempt}/{max_attempts})...")
                            await asyncio.sleep(sleep_time)
                            attempt += 1
                            continue
                        else:
                            response.raise_for_status()

                    response.raise_for_status()
                    return response.json()

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < max_attempts:
                    sleep_time = backoff_base ** attempt
                    print(f"⚠️ [shopify_api] Error de red/timeout ({e}). Reintentando en {sleep_time}s (Intento {attempt}/{max_attempts})...")
                    await asyncio.sleep(sleep_time)
                    attempt += 1
                    continue
                else:
                    raise

    try:
        # Ejecutar primera query para obtener el ProductCategory
        product_res = await post_graphql(product_query, {"id": product_gid})
        
        # Verificar errores dentro del JSON
        if "errors" in product_res and product_res["errors"]:
            print(f"❌ [shopify_api] Errores GraphQL en GetProductCategory: {product_res['errors']}")
            return ("", False)
            
        product_data = product_res.get("data", {}).get("product")
        if not product_data:
            print("⚠️ [shopify_api] Producto no encontrado en Shopify.")
            return ("", False)

        prod_category = product_data.get("productCategory")
        if not prod_category:
            # Producto no tiene categoría asignada en Shopify (Caso C)
            print("ℹ️ [shopify_api] Producto no tiene categoría asignada.")
            return ("", True)

        taxonomy_node = prod_category.get("productTaxonomyNode")
        if not taxonomy_node:
            print("ℹ️ [shopify_api] ProductCategory existe pero no tiene productTaxonomyNode.")
            return ("", True)

        category_id = taxonomy_node.get("id")
        category_fullname = taxonomy_node.get("fullName")

        if not category_id:
            print("⚠️ [shopify_api] productTaxonomyNode no contiene ID.")
            return ("", False)

        # Para consultar atributos y choice lists de la categoría, convertimos el ID a TaxonomyCategory
        # E.g. "gid://shopify/ProductTaxonomyNode/123" -> "gid://shopify/TaxonomyCategory/123" (si aplica)
        taxonomy_category_id = category_id
        if "ProductTaxonomyNode" in category_id:
            taxonomy_category_id = category_id.replace("ProductTaxonomyNode", "TaxonomyCategory")

        # Query 2: Obtener atributos de la categoría
        attributes_query = """
        query GetCategoryAttributes($id: ID!) {
          taxonomyCategory(id: $id) {
            fullName
            attributes(first: 50) {
              edges {
                node {
                  name
                  ... on TaxonomyChoiceListAttribute {
                    choices(first: 50) {
                      edges {
                        node {
                          name
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """

        try:
            attr_res = await post_graphql(attributes_query, {"id": taxonomy_category_id})
            
            # Verificar errores en el JSON
            if "errors" in attr_res and attr_res["errors"]:
                # Si falla la consulta de atributos (por ejemplo, endpoint o permisos),
                # caemos en fallback a retornar al menos la categoría principal
                print(f"⚠️ [shopify_api] Errores GraphQL en GetCategoryAttributes: {attr_res['errors']}")
                return (f"Categoría Shopify: \"{category_fullname}\"", True)

            category_data = attr_res.get("data", {}).get("taxonomyCategory")
            if not category_data:
                # Si no retorna la categoría, retornamos solo el nombre de la categoría
                return (f"Categoría Shopify: \"{category_fullname}\"", True)

            attr_edges = category_data.get("attributes", {}).get("edges", [])
            
            formatted_attrs = []
            for edge in attr_edges:
                node = edge.get("node", {})
                attr_name = node.get("name")
                if not attr_name:
                    continue

                # Si es un TaxonomyChoiceListAttribute, extraemos sus choices
                choices_edges = node.get("choices", {}).get("edges", [])
                if choices_edges:
                    choice_names = [ch.get("node", {}).get("name") for ch in choices_edges if ch.get("node", {}).get("name")]
                    if choice_names:
                        # Limitamos a un máximo de 15 valores en el string para evitar prompts gigantes
                        choice_str = ", ".join(choice_names[:15])
                        if len(choice_names) > 15:
                            choice_str += f" y {len(choice_names) - 15} opciones más"
                        formatted_attrs.append(f"- [{attr_name}]: {choice_str}")
                else:
                    # Atributo simple sin lista de opciones predefinidas
                    formatted_attrs.append(f"- [{attr_name}]")

            prompt_text = f"Categoría Shopify: \"{category_fullname}\""
            if formatted_attrs:
                prompt_text += "\nAtributos requeridos:\n" + "\n".join(formatted_attrs)

            return (prompt_text, True)

        except Exception as attr_err:
            print(f"⚠️ [shopify_api] Error recuperando atributos de categoría: {attr_err}")
            # Retornar al menos la categoría principal en vez de fallar del todo
            return (f"Categoría Shopify: \"{category_fullname}\"", True)

    except Exception as e:
        print(f"❌ [shopify_api] Error en get_product_taxonomy: {e}")
        return ("", False)
