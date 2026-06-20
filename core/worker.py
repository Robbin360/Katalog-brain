import os
import time
import asyncio
from datetime import date, datetime, timedelta, timezone
from typing import Any

import stripe
from core.graph import katalog_agent, supabase
from agents.inspector_agent import inspector_agent

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

PATROL_INTERVAL_SECONDS: int = 30
MAX_PRODUCTS_PER_PATROL: int = 3
STATUS_PROCESSING: str = "PROCESSING"


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


async def auto_pilot_patrol() -> None:
    """
    Definitive background worker for Katalog Auto-Pilot.
    Executes a structured 4-phase cycle:
      - Phase 0: Zombie Sweeper (Global maintenance)
      - Phase 0.5: Auto-Triaje Autónomo (Lead Magnet)
      - Phase 2: Credit Gate (Business filter)
      - Phase 3: Auto-Pilot Optimizer (Heavy optimization)
    """
    print("🤖 [Auto-Pilot] Worker iniciado con arquitectura de 4 fases.")

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
            zombie_timeout = (current_time - timedelta(minutes=15)).strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Buscar productos atascados en PROCESSING
            zombie_res = await _run_sync(
                lambda: supabase.table("shopify_products")
                .select("id,retry_attempts,user_id,billing_state,reservation_id")
                .eq("audit_status", STATUS_PROCESSING)
                .lt("updated_at", zombie_timeout)
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
                            "error_log": "Zombie recovered: PROCESSING timeout (>15 min)",
                            "retry_attempts": nr,
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
        # 🔍 FASE 0.5: Auto-Triaje Autónomo (Lead Magnet - Para todos)
        # ==========================================
        try:
            pending_res = await _run_sync(
                lambda: supabase.table("shopify_products")
                .select("id,current_title,current_body_html")
                .eq("audit_status", "PENDING_AUDIT")
                .limit(3)
                .execute()
            )
            pending_products: list[dict[str, Any]] = pending_res.data or []
            
            if pending_products:
                print(f"🔍 [Auto-Triaje] {len(pending_products)} productos en PENDING_AUDIT detectados para análisis rápido.")
                for idx, prod in enumerate(pending_products):
                    prod_id = prod.get("id")
                    title = prod.get("current_title") or ""
                    body_html = prod.get("current_body_html") or ""
                    
                    if not prod_id:
                        continue
                    
                    print(f"👀 [Auto-Triaje] Evaluando: {title[:30]}...")
                    prompt = f"Title: {title}\nDescription: {body_html}"
                    
                    # Llamamos al inspector_agent de forma asíncrona
                    result = await inspector_agent.run(prompt)
                    audit_data = getattr(result, "data", getattr(result, "output", result))
                    
                    score = _to_int(getattr(audit_data, "score", 0))
                    reason = str(getattr(audit_data, "reason", "Sin justificación."))
                    status = "OPTIMIZED" if score >= 90 else "NEEDS_OPTIMIZATION"
                    
                    # Guardamos el veredicto en Supabase
                    await _run_sync(
                        lambda pid=prod_id, s=score, r=reason, st=status: supabase.table("shopify_products")
                        .update({
                            "seo_score_initial": s,
                            "audit_score": s,
                            "error_log": r,
                            "audit_status": st,
                            "updated_at": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                        }, returning="representation")
                        .eq("id", pid)
                        .execute()
                    )
                    print(f"✅ [Auto-Triaje] Producto ID {prod_id} auditado. Score: {score}/100 | Status: {status}")
                    
                    # Rate Limit Guard: Esperar 15s entre productos si es un lote con más de uno
                    if len(pending_products) > 1 and idx < len(pending_products) - 1:
                        print("⏳ [Auto-Triaje] Enfriando motor por 15 segundos para evitar límite de cuota...")
                        await asyncio.sleep(15)
        except Exception as triage_error:
            print(f"⚠️ [Auto-Triaje] Error durante el triaje autónomo: {triage_error}")

        # ==========================================
        # 🚦 FASE 2: El Peaje de Créditos (Filtro de Negocio)
        # ==========================================
        profiles: list[dict[str, Any]] = []
        try:
            profiles_res = await _run_sync(
                lambda: supabase.table("profiles")
                .select("id,auto_pilot_enabled,credits_used,credits_total,credits_reserved")
                .eq("auto_pilot_enabled", True)
                .execute()
            )
            all_enabled_profiles: list[dict[str, Any]] = profiles_res.data or []
            
            # Filtrar perfiles con créditos disponibles (restando lo ya reservado)
            for profile in all_enabled_profiles:
                credits_used = _to_int(profile.get("credits_used"))
                credits_total = _to_int(profile.get("credits_total"))
                credits_reserved = _to_int(profile.get("credits_reserved"))
                if credits_total - credits_used - credits_reserved > 0:
                    profiles.append(profile)
                    
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
                    credits_used = _to_int(profile.get("credits_used"))
                    credits_total = _to_int(profile.get("credits_total"))
                    credits_reserved = _to_int(profile.get("credits_reserved"))
                    credits_remaining = max(credits_total - credits_used - credits_reserved, 0)
                    
                    if credits_remaining <= 0:
                        continue
                    
                    batch_limit = min(MAX_PRODUCTS_PER_PATROL, credits_remaining)
                    current_iso_time = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
                    
                    eligible_filter = (
                        f"and(audit_status.eq.NEEDS_OPTIMIZATION,retry_attempts.lt.3,or(next_retry_at.lte.{current_iso_time},next_retry_at.is.null)),"
                        f"audit_status.eq.READY_TO_PUBLISH,"
                        f"audit_status.eq.OUT_OF_CREDITS,"
                        f"and(audit_status.eq.ERROR,retry_attempts.lt.3,or(next_retry_at.lte.{current_iso_time},next_retry_at.is.null))"
                    )
                    
                    products_res = await _run_sync(
                        lambda: supabase.table("shopify_products")
                        .select("id,user_id,current_title,audit_status,retry_attempts,next_retry_at")
                        .eq("user_id", user_id)
                        .or_(eligible_filter)
                        .limit(batch_limit)
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
                    
                    for product in claimed_products:
                        product_id = str(product.get("id", ""))
                        if not product_id:
                            continue
                        
                        try:
                            action = "Publicando" if product.get("audit_status") == "READY_TO_PUBLISH" else "Optimizando"
                            print(f"🚀 [Auto-Pilot] {action} producto ID {product_id}...")
                            final_state = await katalog_agent.ainvoke({
                                "product_id": product_id,
                                "auto_pilot_enabled": True,
                                "current_status": product.get("audit_status"),
                            })
                            
                            if final_state.get("error") or final_state.get("status") == "ERROR":
                                print(
                                    f"❌ [Auto-Pilot] Producto ID {product_id} terminó con error: "
                                    f"{final_state.get('error', 'Ver error_log')}"
                                )
                            else:
                                print(f"✅ [Auto-Pilot] Producto ID {product_id} optimizado con éxito.")
                        except Exception as product_error:
                            print(f"❌ [Auto-Pilot] Error procesando producto ID {product_id}: {product_error}")
                            
                except Exception as user_products_error:
                    print(f"⚠️ [Auto-Pilot] Error procesando lote del usuario {user_id}: {user_products_error}")

        # ==========================================
        # ⏳ Enfriamiento de ciclo
        # ==========================================
        print(f"⏳ [Auto-Pilot] Patrullaje completado. Esperando {PATROL_INTERVAL_SECONDS} segundos...")
        try:
            await asyncio.sleep(PATROL_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            print("🛑 [Auto-Pilot] Worker detenido por apagado del servidor.")
            raise
