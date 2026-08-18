# scripts/inspect_all_products.py
"""Recorre todos los productos y reporta qué datos enriquecibles existen en Shopify.

Objetivo: decidir con datos si vale la pena enriquecer el sync. Verifica cuántos
productos traen productType, sku real, barcode y metafields.

Costo: ~8 puntos de rate limit por producto (saldo 20.000, recarga 1000/s).
"""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

API_VERSION = "2026-04"

QUERY = """
query($ids: [ID!]!) {
  nodes(ids: $ids) {
    ... on Product {
      id
      title
      productType
      category { fullName }
      variants(first: 5) {
        edges { node { sku barcode } }
      }
      metafields(first: 30) {
        edges { node { namespace key value type } }
      }
    }
  }
}
"""


async def main() -> None:
    s = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])

    rows = (
        s.table("shopify_products")
        .select("id, user_id, shopify_id")
        .order("id")
        .execute()
        .data
    )
    if not rows:
        print("No hay productos.")
        return

    user_id = rows[0]["user_id"]
    integ = (
        s.table("integrations")
        .select("shop_url, access_token")
        .eq("user_id", user_id)
        .eq("provider", "shopify")
        .limit(1)
        .execute()
        .data[0]
    )
    token = s.rpc("decrypt_shopify_token", {"p_ciphertext_b64": integ["access_token"]}).execute().data

    by_gid = {r["shopify_id"]: r["id"] for r in rows}
    gids = list(by_gid.keys())
    url = f"https://{integ['shop_url']}/admin/api/{API_VERSION}/graphql.json"
    headers = {"Content-Type": "application/json", "X-Shopify-Access-Token": token}

    nodes = []
    async with httpx.AsyncClient(timeout=45.0) as client:
        for i in range(0, len(gids), 10):
            chunk = gids[i : i + 10]
            r = await client.post(
                url, json={"query": QUERY, "variables": {"ids": chunk}}, headers=headers
            )
            data = r.json()
            if data.get("errors"):
                print("ERRORES GraphQL:", json.dumps(data["errors"], indent=2, ensure_ascii=False))
                return
            nodes.extend(n for n in data["data"]["nodes"] if n)
            cost = data.get("extensions", {}).get("cost", {})
            print(f"  lote {i // 10 + 1}: costo {cost.get('actualQueryCost')} pts")

    print("\n" + "=" * 100)
    con_type = con_sku = con_barcode = con_meta = con_cat = 0
    namespaces: dict[str, int] = {}

    for n in sorted(nodes, key=lambda x: by_gid.get(x["id"], 0)):
        pid = by_gid.get(n["id"], "?")
        ptype = n.get("productType") or ""
        cat = (n.get("category") or {}).get("fullName") or ""
        variants = [e["node"] for e in n.get("variants", {}).get("edges", [])]
        skus = [v.get("sku") for v in variants if v.get("sku")]
        barcodes = [v.get("barcode") for v in variants if v.get("barcode")]
        metas = [e["node"] for e in n.get("metafields", {}).get("edges", [])]

        if ptype:
            con_type += 1
        if skus:
            con_sku += 1
        if barcodes:
            con_barcode += 1
        if cat:
            con_cat += 1
        if metas:
            con_meta += 1
        for m in metas:
            key = f"{m['namespace']}.{m['key']}"
            namespaces[key] = namespaces.get(key, 0) + 1

        print(f"\n[{pid}] {(n.get('title') or '')[:55]}")
        print(f"     type={ptype or '-'} | cat={cat or '-'} | variantes={len(variants)}")
        print(f"     sku={skus or '-'} | barcode={barcodes or '-'}")
        if metas:
            for m in metas:
                val = str(m.get("value"))[:70]
                print(f"     META {m['namespace']}.{m['key']} ({m.get('type')}) = {val}")
        else:
            print("     META (ninguno)")

    total = len(nodes)
    print("\n" + "=" * 100)
    print(f"TOTAL {total} productos")
    print(f"  con productType : {con_type}")
    print(f"  con categoría   : {con_cat}")
    print(f"  con sku         : {con_sku}")
    print(f"  con barcode     : {con_barcode}")
    print(f"  con metafields  : {con_meta}")
    if namespaces:
        print("\n  Claves de metafields encontradas:")
        for k, v in sorted(namespaces.items(), key=lambda kv: -kv[1]):
            print(f"    {k}: {v} productos")


if __name__ == "__main__":
    asyncio.run(main())
