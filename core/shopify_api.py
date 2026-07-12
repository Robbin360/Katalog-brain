import asyncio
from datetime import datetime, timezone
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


# --- SHOPIFY PAGINATION + THROTTLE + RESUME + UPSERT (FIX 16) ---

async def shopify_graphql_request(shop_url: str, access_token: str, query: str, variables: dict = None) -> dict:
    """Realiza una petición GraphQL a la API Admin de Shopify."""
    normalized_shop_url = _normalize_shop_url(shop_url)
    endpoint = f"https://{normalized_shop_url}/admin/api/{SHOPIFY_API_VERSION}/graphql.json"
    headers = {
        "Content-Type": "application/json",
        "X-Shopify-Access-Token": access_token,
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(
            endpoint,
            json={"query": query, "variables": variables or {}},
            headers=headers
        )
        response.raise_for_status()
        return response.json()


async def handle_throttle(throttle_status: dict):
    """Pausa si estamos cerca del rate limit de Shopify."""
    if not throttle_status:
        return
    
    currently_available = throttle_status.get("currentlyAvailable", 1000)
    restore_rate = throttle_status.get("restoreRate", 50)
    
    # Si nos quedan menos de 250 puntos (costo de la próxima query), esperar
    if currently_available < 250:
        points_needed = 250 - currently_available
        wait_seconds = points_needed / max(restore_rate, 1)
        wait_seconds = wait_seconds * 1.1  # 10% margen
        print(f"⏳ [Throttle] Waiting {wait_seconds:.2f} seconds for Shopify rate limit points to restore.")
        await asyncio.sleep(min(wait_seconds, 10))  # Máximo 10s de espera


async def update_sync_cursor(job_id: str, cursor: str, products_synced: int):
    """Guarda el cursor y el progreso después de cada página."""
    from core.graph import supabase
    
    def _update():
        supabase.table("sync_jobs").update({
            "last_sync_cursor": cursor,
            "products_synced": products_synced,
            "status": "syncing"
        }).eq("id", job_id).execute()
    
    await asyncio.to_thread(_update)


async def complete_sync_job(job_id: str, products_synced: int):
    """Marca el sync como completo y limpia el cursor."""
    from core.graph import supabase
    
    def _update():
        supabase.table("sync_jobs").update({
            "status": "completed",
            "products_synced": products_synced,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "last_sync_cursor": None
        }).eq("id", job_id).execute()
    
    await asyncio.to_thread(_update)


async def fail_sync_job(job_id: str, error_message: str):
    """Marca el sync como fallido."""
    from core.graph import supabase
    
    def _update():
        supabase.table("sync_jobs").update({
            "status": "failed",
            "error_message": error_message,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", job_id).execute()
    
    await asyncio.to_thread(_update)


async def fetch_all_products(shop_url: str, access_token: str, job_id: str, last_cursor: str = None, initial_count: int = 0):
    """Trae TODOS los productos de una tienda Shopify usando cursor-based pagination y actualiza el job.
    initial_count es el conteo previo para resume (productos ya sincronizados antes de la interrupción)."""
    all_products = []
    cursor = last_cursor  # None si es sync desde cero, o el último cursor guardado si es resume
    has_next = True
    
    while has_next:
        query = """
        query getProducts($first: Int!, $after: String) {
          products(first: $first, after: $after) {
            edges {
              node {
                id
                title
                descriptionHtml
                seo {
                  title
                  description
                }
                status
                vendor
                productType
                tags
                featuredImage {
                  url
                }
                variants(first: 10) {
                  edges {
                    node {
                      id
                      sku
                      price
                      inventoryQuantity
                    }
                  }
                }
              }
              cursor
            }
            pageInfo {
              hasNextPage
              endCursor
            }
          }
        }
        """
        
        variables = {
            "first": 250,
            "after": cursor
        }
        
        response = await shopify_graphql_request(
            shop_url, 
            access_token, 
            query, 
            variables
        )
        
        if "errors" in response and response["errors"]:
            raise RuntimeError(f"Shopify GraphQL Error: {response['errors'][0].get('message')}")
            
        products_page = response["data"]["products"]["edges"]
        page_info = response["data"]["products"]["pageInfo"]
        
        for edge in products_page:
            product = edge["node"]
            all_products.append(product)
        
        # Actualizar cursor
        cursor = page_info["endCursor"]
        has_next = page_info["hasNextPage"]
        
        # Guardar el cursor en sync_jobs después de CADA página (para resume)
        # Usar conteo TOTAL (previos + nuevos de esta sesión)
        total_so_far = initial_count + len(all_products)
        await update_sync_cursor(job_id, cursor, total_so_far)
        
        # Throttle: leer el costo de la query del response y pacear
        throttle_status = response.get("extensions", {}).get("cost", {}).get("throttleStatus", {})
        await handle_throttle(throttle_status)
    
    return all_products


async def save_products_to_db(products: list, user_id: str):
    """Guarda productos con upsert usando (shopify_id, user_id) como conflict key."""
    from core.graph import supabase
    
    records = []
    for product in products:
        # Extraer variantes
        variants_edges = product.get("variants", {}).get("edges", [])
        first_variant = variants_edges[0].get("node", {}) if variants_edges else {}
        price_str = first_variant.get("price")
        price_val = float(price_str) if price_str is not None else None
        
        tags = product.get("tags", [])
        tags_str = ", ".join(tags) if isinstance(tags, list) else str(tags or "")
        
        record = {
            "shopify_id": product["id"],  # gid://shopify/Product/...
            "user_id": user_id,
            "current_title": product.get("title", ""),
            "current_body_html": product.get("descriptionHtml", ""),
            "vendor": product.get("vendor", ""),
            "tags": tags_str,
            "image_url": product.get("featuredImage", {}).get("url", "") if product.get("featuredImage") else "",
            "price": price_val,
            "inventory_quantity": first_variant.get("inventoryQuantity", 0) if first_variant else 0,
            "audit_status": "PENDING_AUDIT",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        records.append(record)
    
    # Upsert en lotes de 100
    batch_size = 100
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        
        def _upsert(b=batch):
            supabase.table("shopify_products").upsert(
                b,
                on_conflict="shopify_id,user_id"
            ).execute()
        
        await asyncio.to_thread(_upsert)


async def sync_shopify_products(user_id: str, shop_url: str, access_token: str):
    """Sync completo con resume."""
    from core.graph import supabase
    
    # 1. Buscar si hay un sync_job en progreso (para resume)
    def _find_prev_job():
        return supabase.table("sync_jobs")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("status", "syncing")\
            .order("created_at", desc=True)\
            .limit(1)\
            .execute()
    
    result = await asyncio.to_thread(_find_prev_job)
    
    if result.data:
        # RESUME: hay un sync interrumpido — continuar desde el cursor
        job = result.data[0]
        job_id = job["id"]
        last_cursor = job.get("last_sync_cursor")
        already_synced = job.get("products_synced", 0)
        print(f"Resuming sync from cursor: {last_cursor} (already synced: {already_synced})")
        products = await fetch_all_products(shop_url, access_token, job_id, last_cursor=last_cursor, initial_count=already_synced)
        total_synced = already_synced + len(products)
    else:
        # FRESH: nuevo sync desde cero
        # Obtener total aproximado antes de empezar
        products_total = 0
        try:
            count_res = await shopify_graphql_request(
                shop_url,
                access_token,
                "query { productsCount { count } }"
            )
            products_total = count_res.get("data", {}).get("productsCount", {}).get("count", 0)
        except Exception as count_err:
            print(f"⚠️ Could not fetch productsCount: {count_err}")
            
        # No hay sync en progreso — crear nuevo job
        def _create_job():
            return supabase.table("sync_jobs").insert({
                "user_id": user_id,
                "status": "syncing",
                "products_total": products_total,
                "started_at": datetime.now(timezone.utc).isoformat()
            }).execute()
        
        job_result = await asyncio.to_thread(_create_job)
        job_id = job_result.data[0]["id"]
        print("Starting fresh sync")
        products = await fetch_all_products(shop_url, access_token, job_id, last_cursor=None, initial_count=0)
        total_synced = len(products)
    
    # 2. Guardar productos en shopify_products (upsert)
    await save_products_to_db(products, user_id)
    
    # 3. Marcar sync como completo con el conteo TOTAL
    await complete_sync_job(job_id, total_synced)
    
    return {"status": "completed", "products_synced": total_synced}
