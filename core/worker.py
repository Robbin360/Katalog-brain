import asyncio
from typing import Any

from core.graph import katalog_agent, supabase


PATROL_INTERVAL_SECONDS: int = 30
MAX_PRODUCTS_PER_PATROL: int = 3
STATUS_PENDING_AUDIT: str = "PENDING_AUDIT"
STATUS_PROCESSING: str = "PROCESSING"
STATUS_OUT_OF_CREDITS: str = "OUT_OF_CREDITS"


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


async def _run_sync(callable_obj):
    return await asyncio.to_thread(callable_obj)


async def _claim_pending_products(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected_ids = [product["id"] for product in products if product.get("id")]
    if not selected_ids:
        return []

    claim_res = await _run_sync(
        lambda: supabase.table("shopify_products")
        .update({"audit_status": STATUS_PROCESSING})
        .in_("id", selected_ids)
        .eq("audit_status", STATUS_PENDING_AUDIT)
        .select("id")
        .execute()
    )
    claimed_ids = {str(product.get("id")) for product in (claim_res.data or [])}

    if not claimed_ids:
        print("⚠️ [Auto-Pilot] Ningún producto fue reservado. Otro worker pudo tomar el lote.")
        return []

    return [product for product in products if str(product.get("id")) in claimed_ids]


async def auto_pilot_patrol() -> None:
    """
    Background worker for Katalog Auto-Pilot.
    Patrols Supabase for enabled users and processes pending product audits.
    """
    print("🤖 [Auto-Pilot] Worker iniciado. Patrullando Supabase cada 30 segundos.")

    while True:
        try:
            print("⏳ [Auto-Pilot] Buscando usuarios con Auto-Pilot activo...")

            profiles_res = await _run_sync(
                lambda: supabase.table("profiles")
                .select("id,auto_pilot_enabled,credits_used,credits_total")
                .eq("auto_pilot_enabled", True)
                .execute()
            )
            profiles: list[dict[str, Any]] = profiles_res.data or []

            if not profiles:
                print("✅ [Auto-Pilot] No hay usuarios activos para patrullar.")
            else:
                print(f"🚀 [Auto-Pilot] Usuarios activos detectados: {len(profiles)}")

                for profile in profiles:
                    user_id = str(profile.get("id", ""))
                    if not user_id:
                        print("⚠️ [Auto-Pilot] Perfil sin ID. Saltando usuario.")
                        continue

                    credits_used = _to_int(profile.get("credits_used"))
                    credits_total = _to_int(profile.get("credits_total"))
                    credits_remaining = max(credits_total - credits_used, 0)

                    if credits_remaining <= 0:
                        print(f"🛑 [Auto-Pilot] Usuario {user_id} sin créditos. Marcando pendientes.")
                        await _run_sync(lambda: supabase.table("shopify_products").update({
                            "audit_status": STATUS_OUT_OF_CREDITS
                        }).eq("user_id", user_id).eq("audit_status", STATUS_PENDING_AUDIT).execute())
                        continue

                    batch_limit = min(MAX_PRODUCTS_PER_PATROL, credits_remaining)
                    products_res = await _run_sync(
                        lambda: supabase.table("shopify_products")
                        .select("id,user_id,current_title,audit_status")
                        .eq("user_id", user_id)
                        .eq("audit_status", STATUS_PENDING_AUDIT)
                        .limit(batch_limit)
                        .execute()
                    )
                    products: list[dict[str, Any]] = products_res.data or []

                    if not products:
                        print(f"✅ [Auto-Pilot] Usuario {user_id} sin productos pendientes.")
                    else:
                        claimed_products = await _claim_pending_products(products)
                        if not claimed_products:
                            continue

                        print(
                            "🤖 [Auto-Pilot] "
                            f"Usuario {user_id}: {len(claimed_products)} productos reservados "
                            f"({credits_remaining} créditos disponibles)."
                        )

                        for product in claimed_products:
                            product_id = str(product.get("id", ""))
                            if not product_id:
                                print("⚠️ [Auto-Pilot] Producto sin ID. Saltando registro.")
                                continue

                            try:
                                print(f"🚀 [Auto-Pilot] Optimizando producto ID {product_id}...")
                                final_state = await katalog_agent.ainvoke({
                                    "product_id": product_id,
                                    "auto_pilot_enabled": True
                                })

                                if final_state.get("error") or final_state.get("status") == "ERROR":
                                    print(
                                        "❌ [Auto-Pilot] Producto "
                                        f"ID {product_id} terminó con error: "
                                        f"{final_state.get('error', 'Ver error_log')}"
                                    )
                                else:
                                    print(f"✅ [Auto-Pilot] Producto ID {product_id} procesado.")
                            except Exception as product_error:
                                print(
                                    "❌ [Auto-Pilot] Error procesando producto "
                                    f"ID {product_id}: {product_error}"
                                )

        except asyncio.CancelledError:
            print("🛑 [Auto-Pilot] Worker detenido por apagado del servidor.")
            raise
        except Exception as patrol_error:
            print(f"❌ [Auto-Pilot] Error general del patrullaje: {patrol_error}")

        print(f"⏳ [Auto-Pilot] Esperando {PATROL_INTERVAL_SECONDS} segundos...")
        try:
            await asyncio.sleep(PATROL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            print("🛑 [Auto-Pilot] Worker detenido por apagado del servidor.")
            raise
