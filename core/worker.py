import asyncio
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

import stripe
from core.graph import katalog_agent, supabase
from core.helpers import utc_now_iso

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

DEFAULT_PATROL_INTERVAL_SECONDS: int = 60

TIER_PATROL_INTERVALS = {
    "free": 60,
    "starter": 60,
    "pro": 300,
    "business": 600,
    "enterprise": 600,
}

DEFAULT_BATCH_SIZE: int = 3
MAX_BATCH_SIZE_CAP: int = 50

TIER_BATCH_PARALLELISM = {
    "free": 1,
    "starter": 1,
    "pro": 2,
    "business": 3,
    "enterprise": 5,
}

STATUS_PROCESSING: str = "PROCESSING"

# 45 min, no 15: una corrida real del 1010 tardó 21.8 minutos. Con latido por
# nodo (core/graph.py:_heartbeat), 45 minutos sin latir sí significa muerto.
ZOMBIE_TIMEOUT_MINUTES: int = 45


def _to_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


async def _run_sync(callable_obj):
    return await asyncio.to_thread(callable_obj)


async def _claim_retryable_products(
    products: list[dict[str, Any]],
    eligible_filter: str,
) -> list[dict[str, Any]]:
    selected_ids = [product["id"] for product in products if product.get("id")]
    if not selected_ids:
        return []

    claim_res = await _run_sync(
        lambda: supabase.table("shopify_products")
        .update({
            "audit_status": STATUS_PROCESSING,
            "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        }, returning="representation")
        .in_("id", selected_ids)
        .or_(eligible_filter)
        .execute()
    )
    claimed_ids = {str(product.get("id")) for product in (claim_res.data or []) if product.get("id")}

    if not claimed_ids:
        claim_window_start = (
            datetime.now(timezone.utc) - timedelta(seconds=30)
        ).strftime('%Y-%m-%dT%H:%M:%SZ')
        confirm_res = await _run_sync(
            lambda: supabase.table("shopify_products")
            .select("id")
            .in_("id", selected_ids)
            .eq("audit_status", STATUS_PROCESSING)
            .gte("updated_at", claim_window_start)
            .execute()
        )
        claimed_ids = {
            str(product.get("id"))
            for product in (confirm_res.data or [])
            if product.get("id")
        }
        if not claimed_ids:
            print("ℹ️ [Auto-Pilot] Claim sin filas confirmadas. Otro worker pudo ganar la carrera; lote omitido.")
            return []

    return [product for product in products if str(product.get("id")) in claimed_ids]


def _reconcile_stripe_subscriptions() -> int:
    past_due_res = supabase.table("profiles").select(
        "id,stripe_subscription_id,subscription_status"
    ).eq("subscription_status", "past_due").execute()
    past_due_users: list[dict[str, Any]] = past_due_res.data or []

    if not past_due_users:
        return 0

    reconciled = 0
    for i in range(0, len(past_due_users), 10):
        batch = past_due_users[i:i + 10]
        for user in batch:
            sub_id = user.get("stripe_subscription_id")
            if not sub_id:
                continue
            try:
                subscription = stripe.Subscription.retrieve(sub_id)
                status = subscription.status
            except Exception as e:
                print(f"⚠️ [Stripe Sync] Error consultando suscripción {sub_id} del usuario {user.get('id')}: {e}")
                continue

            user_id = user.get("id")
            if status in ("canceled", "unpaid"):
                supabase.table("profiles").update({
                    "subscription_status": "inactive",
                    "plan_tier": "free",
                    "auto_pilot_enabled": False,
                    "credits_total": 0,
                }).eq("id", user_id).execute()
                reconciled += 1
            elif status == "active":
                supabase.table("profiles").update({
                    "subscription_status": "active",
                    "auto_pilot_enabled": True,
                }).eq("id", user_id).execute()
                reconciled += 1

        if i + 10 < len(past_due_users):
            time.sleep(0.2)

    return reconciled


# Global de control para ejecutar la reconciliación 1 vez por día
_last_reconciliation_date: date | None = None


async def process_single_product(product: dict, user_id: str, plan_tier: str) -> dict:
    product_id = str(product.get("id", ""))
    if not product_id:
        return {"product_id": None, "success": False, "error": "missing_id", "credits_consumed": 0}

    try:
        action = "Publicando" if product.get("audit_status") == "READY_TO_PUBLISH" else "Optimizando"
        print(f"🚀 [Auto-Pilot] {action} producto ID {product_id} (user: {user_id}, tier: {plan_tier})...")

        final_state = await katalog_agent.ainvoke({
            "product_id": product_id,
            "auto_pilot_enabled": True,
            "current_status": product.get("audit_status"),
        })

        if final_state.get("error") or final_state.get("status") == "ERROR":
            error_msg = final_state.get("error", "unknown_error")
            print(f"❌ [Auto-Pilot] Producto ID {product_id} terminó con error: {error_msg}")
            return {
                "product_id": product_id,
                "success": False,
                "error": error_msg,
                "credits_consumed": 0,
            }
        else:
            print(f"✅ [Auto-Pilot] Producto ID {product_id} optimizado con éxito.")
            return {
                "product_id": product_id,
                "success": True,
                "error": None,
                "credits_consumed": 1,
            }

    except Exception as product_error:
        print(f"❌ [Auto-Pilot] Error procesando producto ID {product_id}: {product_error}")
        return {
            "product_id": product_id,
            "success": False,
            "error": str(product_error),
            "credits_consumed": 0,
        }


async def auto_pilot_patrol() -> None:
    """
    Definitive background worker for Katalog Auto-Pilot.
    Executes a structured 4-phase cycle:
      - Phase 0: Zombie Sweeper (Global maintenance)
      - Phase 2: Credit Gate (Business filter)
      - Phase 3: Auto-Pilot Optimizer (Heavy optimization)
    """
    print("🤖 [Auto-Pilot] Worker iniciado con arquitectura de fases.")

    while True:
        # ==========================================
        # 💵 FASE -1: Reconciliación Diaria de Pagos (Stripe Sync)
        # ==========================================
        global _last_reconciliation_date
        try:
            today = datetime.now(timezone.utc).date()
            if _last_reconciliation_date != today:
                reconciled_count = await _run_sync(_reconcile_stripe_subscriptions)
                print(f"🔍💵 [Stripe Sync] {reconciled_count} usuarios reconciliados con Stripe.")
                _last_reconciliation_date = today
        except Exception as stripe_error:
            print(f"⚠️ [Stripe Sync] Error durante reconciliación: {stripe_error}")

        # ==========================================
        # 🧟 FASE 0: Zombie Sweeper (Mantenimiento Global - Sin costo)
        # ==========================================
        try:
            current_time = datetime.now(timezone.utc)
            zombie_timeout = (current_time - timedelta(minutes=ZOMBIE_TIMEOUT_MINUTES)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Buscar productos atascados en PROCESSING. Dos casos: latido vencido,
            # o latido NULL (filas previas a este cambio o proceso muerto antes del
            # primer latido) con updated_at vencido — un .lt() sobre NULL nunca
            # coincide en Postgres y esas filas quedarían huérfanas para siempre.
            zombie_res = await _run_sync(
                lambda zt=zombie_timeout: supabase.table("shopify_products")
                .select("id,retry_attempts,user_id,billing_state,reservation_id,processing_heartbeat_at")
                .eq("audit_status", STATUS_PROCESSING)
                .or_(f"processing_heartbeat_at.lt.{zt},and(processing_heartbeat_at.is.null,updated_at.lt.{zt})")
                .execute()
            )
            zombies: list[dict[str, Any]] = zombie_res.data or []
            
            if zombies:
                print(f"🔎 [DEBUG] Zombis crudos de la DB: {zombies}")
                print(f"🧟 [Zombie Sweeper] {len(zombies)} zombies detectados en PROCESSING. Rescatando...")
                for zombie in zombies:
                    zombie_id = zombie.get("id")
                    if not zombie_id:
                        continue
                    new_retry = _to_int(zombie.get("retry_attempts")) + 1
                    
                    update_res = await _run_sync(
                        lambda zid=zombie_id, nr=new_retry: supabase.table("shopify_products")
                        .update({
                            "audit_status": "ERROR",
                            "error_log": f"Zombie recovered: PROCESSING timeout ({ZOMBIE_TIMEOUT_MINUTES} min)",
                            "retry_attempts": nr,
                            "processing_heartbeat_at": None,
                            "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                        }, returning="representation")
                        .eq("id", zid)
                        .execute()
                    )
                    print(f"🔎 [DEBUG] Update resultado para ID {zombie_id}: {update_res.data}")

                    # 💳 Si el zombie aún tenía una reserva viva (nunca llegó al commit),
                    # liberamos el crédito atrapado. Si ya estaba COMMITTED, no se toca.
                    billing_state = zombie.get("billing_state")
                    reservation_id = zombie.get("reservation_id")
                    user_id = zombie.get("user_id")
                    if billing_state == "RESERVED" and reservation_id and user_id:
                        try:
                            await _run_sync(
                                lambda uid=user_id, zid=zombie_id, rid=reservation_id: supabase.rpc(
                                    "refund_product_reservation", {
                                        "p_user_id": uid,
                                        "p_product_id": zid,
                                        "p_reservation_id": rid,
                                    }
                                ).execute()
                            )
                            print(f"↩️ [Zombie Sweeper] Crédito reembolsado para producto {zombie_id}.")
                        except Exception as refund_error:
                            print(f"⚠️ [Zombie Sweeper] Error reembolsando crédito de {zombie_id}: {refund_error}")
                print(f"🧟 [Zombie Sweeper] {len(zombies)} zombies devueltos a la cola de errores.")
        except Exception as sweeper_error:
            print(f"⚠️ [Zombie Sweeper] Error durante la fase de recuperación: {sweeper_error}")

        # ==========================================
        # 🚦 FASE 2: El Peaje de Créditos (Filtro de Negocio)
        # ==========================================
        profiles: list[dict[str, Any]] = []
        try:
            profiles_res = await _run_sync(
                lambda: supabase.table("profile_credits")
                .select("id,auto_pilot_enabled,credits_available,plan_tier,auto_pilot_patrol_limit,feature_flags")
                .eq("auto_pilot_enabled", True)
                .gt("credits_available", 0)
                .execute()
            )
            profiles = profiles_res.data or []

            if not profiles:
                print("💤 [Auto-Pilot] No hay usuarios Pro/Business activos con créditos disponibles.")
            else:
                print(f"🚀 [Auto-Pilot] {len(profiles)} usuarios Pro/Business activos con créditos detectados.")
        except Exception as credit_error:
            print(f"⚠️ [Peaje Créditos] Error comprobando créditos de usuarios: {credit_error}")

        # ==========================================
        # 🚀 FASE 3: El Auto-Pilot (La Cirugía de Pago - Pro/Business)
        # ==========================================
        if profiles:
            for profile in profiles:
                user_id = str(profile.get("id", ""))
                if not user_id:
                    continue
                
                try:
                    credits_remaining = max(_to_int(profile.get("credits_available")), 0)
                    
                    if credits_remaining <= 0:
                        continue
                    
                    plan_tier = profile.get("plan_tier", "free")
                    parallelism = TIER_BATCH_PARALLELISM.get(plan_tier, 1)
                    patrol_interval = TIER_PATROL_INTERVALS.get(plan_tier, DEFAULT_PATROL_INTERVAL_SECONDS)
                    
                    raw_patrol_limit = profile.get("auto_pilot_patrol_limit")
                    try:
                        patrol_limit = int(raw_patrol_limit) if raw_patrol_limit is not None else DEFAULT_BATCH_SIZE
                        patrol_limit = max(1, min(patrol_limit, MAX_BATCH_SIZE_CAP))
                    except (ValueError, TypeError):
                        patrol_limit = DEFAULT_BATCH_SIZE
                    
                    batch_limit = min(patrol_limit, credits_remaining)
                    current_iso_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    
                    eligible_filter = (
                        f"and(audit_status.eq.NEEDS_OPTIMIZATION,retry_attempts.lt.3,or(next_retry_at.lte.{current_iso_time},next_retry_at.is.null)),"
                        f"audit_status.eq.READY_TO_PUBLISH,"
                        f"audit_status.eq.OUT_OF_CREDITS,"
                        f"and(audit_status.eq.ERROR,retry_attempts.lt.3,or(next_retry_at.lte.{current_iso_time},next_retry_at.is.null))"
                    )
                    
                    products_res = await _run_sync(
                        lambda uid=user_id, ef=eligible_filter, bl=batch_limit: supabase.table("shopify_products")
                        .select("id,user_id,current_title,audit_status,retry_attempts,next_retry_at")
                        .eq("user_id", uid)
                        .or_(ef)
                        .limit(bl)
                        .execute()
                    )
                    products: list[dict[str, Any]] = products_res.data or []
                    
                    if not products:
                        continue
                    
                    # Claim Atómico (Fase de reserva a PROCESSING)
                    claimed_products = await _claim_retryable_products(products, eligible_filter)
                    if not claimed_products:
                        continue
                    
                    print(
                        f"🤖 [Auto-Pilot] Usuario {user_id}: {len(claimed_products)} productos reservados "
                        f"({credits_remaining} créditos disponibles)."
                    )
                    
                    patrol_started_at = datetime.now(timezone.utc)

                    products_processed = 0
                    products_succeeded = 0
                    products_failed = 0
                    credits_consumed = 0

                    if parallelism <= 1 or len(claimed_products) <= 1:
                        for product in claimed_products:
                            result = await process_single_product(product, user_id, plan_tier)
                            if isinstance(result, dict):
                                products_processed += 1
                                if result.get("success"):
                                    products_succeeded += 1
                                    credits_consumed += result.get("credits_consumed", 0)
                                else:
                                    products_failed += 1
                    else:
                        print(f"🔄 [Auto-Pilot] Procesando {len(claimed_products)} productos en paralelo (batch_size={parallelism}, tier={plan_tier})...")

                        batches = [
                            claimed_products[i:i + parallelism]
                            for i in range(0, len(claimed_products), parallelism)
                        ]

                        for batch_idx, batch in enumerate(batches):
                            print(f"📦 [Auto-Pilot] Procesando batch {batch_idx + 1}/{len(batches)} ({len(batch)} productos)...")

                            tasks = [
                                process_single_product(product, user_id, plan_tier)
                                for product in batch
                            ]

                            results = await asyncio.gather(*tasks, return_exceptions=True)

                            for result in results:
                                if isinstance(result, Exception):
                                    print(f"❌ [Auto-Pilot] Batch exception: {result}")
                                    products_failed += 1
                                elif isinstance(result, dict):
                                    products_processed += 1
                                    if result.get("success"):
                                        products_succeeded += 1
                                        credits_consumed += result.get("credits_consumed", 0)
                                    else:
                                        products_failed += 1

                        print(
                            f"📊 [Auto-Pilot] Patrol completo para user {user_id}: "
                            f"{products_processed} procesados, {products_succeeded} exitosos, "
                            f"{products_failed} fallidos, {credits_consumed} créditos consumidos"
                        )

                    patrol_completed_at = datetime.now(timezone.utc)
                    try:
                        await _run_sync(
                            lambda uid=user_id, psa=patrol_started_at, pca=patrol_completed_at, pt=plan_tier, pl=patrol_limit, pi=patrol_interval, pp=products_processed, ps=products_succeeded, pf=products_failed, cc=credits_consumed, par=parallelism, cp=claimed_products: supabase.table("auto_pilot_patrol_logs").insert({
                                "user_id": uid,
                                "patrol_started_at": psa.isoformat(),
                                "patrol_completed_at": pca.isoformat(),
                                "plan_tier": pt,
                                "patrol_limit": pl,
                                "patrol_interval_seconds": pi,
                                "feature_flag_enabled": True,
                                "products_processed": pp,
                                "products_succeeded": ps,
                                "products_failed": pf,
                                "credits_consumed": cc,
                                "llm_provider_used": "mixed",
                                "llm_calls_succeeded": 0,
                                "llm_calls_failed": 0,
                                "quota_errors": 0,
                                "error_message": None,
                                "metadata": {
                                    "parallelism": par,
                                    "batch_size": par,
                                    "products_claimed": len(cp),
                                },
                            }).execute()
                        )
                    except Exception as log_error:
                        print(f"⚠️ [Auto-Pilot] Failed to log patrol: {log_error}")

                except Exception as user_products_error:
                    print(f"⚠️ [Auto-Pilot] Error procesando lote del usuario {user_id}: {user_products_error}")

        # ==========================================
        # ⏳ Enfriamiento de ciclo
        # ==========================================
        print(f"⏳ [Auto-Pilot] Patrullaje completado. Esperando {DEFAULT_PATROL_INTERVAL_SECONDS} segundos...")
        try:
            await asyncio.sleep(DEFAULT_PATROL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            print("🛑 [Auto-Pilot] Worker detenido por apagado del servidor.")
            raise
