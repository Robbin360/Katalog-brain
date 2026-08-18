# scripts/inspect_shopify_product.py
import asyncio, json, os
import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

PRODUCT_ID = 1010
API_VERSION = "2026-04"

QUERY = """
query($id: ID!) {
  product(id: $id) {
    title
    productType
    category { fullName }
    variants(first: 20) {
      edges { node { title sku barcode price inventoryQuantity } }
    }
    metafields(first: 20) {
      edges { node { namespace key value } }
    }
  }
}
"""


async def main() -> None:
    s = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    prod = s.table("shopify_products").select("user_id, shopify_id").eq("id", PRODUCT_ID).single().execute().data
    integ = (
        s.table("integrations")
        .select("shop_url, access_token")
        .eq("user_id", prod["user_id"])
        .eq("provider", "shopify")
        .limit(1)
        .execute()
        .data[0]
    )
    token = s.rpc("decrypt_shopify_token", {"p_ciphertext_b64": integ["access_token"]}).execute().data

    url = f"https://{integ['shop_url']}/admin/api/{API_VERSION}/graphql.json"
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            url,
            json={"query": QUERY, "variables": {"id": prod["shopify_id"]}},
            headers={"Content-Type": "application/json", "X-Shopify-Access-Token": token},
        )
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
