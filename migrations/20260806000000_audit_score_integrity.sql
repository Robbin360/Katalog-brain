-- =============================================================
-- Estado REALMENTE DESPLEGADO al 2026-08-08.
-- Este archivo refleja la base, no una propuesta.
-- Verificado con pg_get_functiondef / pg_get_triggerdef.
-- NO editar sin volver a verificar contra la base.
-- =============================================================

-- B1: dos triggers separados sobre trigger_auto_audit(), que NO se modifica.
-- La función encola en products_queue y es el motor del autopilot.
-- OLD no existe en INSERT, por eso van separados.
drop trigger if exists check_product_health on public.shopify_products;

create trigger check_product_health_ins
after insert on public.shopify_products
for each row
when (
    coalesce(new.audit_status, '') not in ('OPTIMIZED','READY_TO_PUBLISH','PROCESSING')
    and coalesce(new.consecutive_failures, 0) < 3
)
execute function public.trigger_auto_audit();

create trigger check_product_health_upd
after update of sales_last_7_days, inventory_quantity, current_title, current_body_html
on public.shopify_products
for each row
when (
    coalesce(new.audit_status, '') not in ('OPTIMIZED','READY_TO_PUBLISH','PROCESSING')
    and coalesce(new.consecutive_failures, 0) < 3
    and (
        new.current_title        is distinct from old.current_title
        or new.current_body_html is distinct from old.current_body_html
        or new.inventory_quantity is distinct from old.inventory_quantity
        or new.sales_last_7_days  is distinct from old.sales_last_7_days
    )
)
execute function public.trigger_auto_audit();

-- B2: firma RETURNS trigger, sin parámetros. SECURITY DEFINER + search_path.
CREATE OR REPLACE FUNCTION public.refresh_user_kpis()
 RETURNS trigger
 LANGUAGE plpgsql
 SECURITY DEFINER
 SET search_path TO 'public', 'pg_temp'
AS $function$
DECLARE
  v_target_user uuid;
  v_revenue numeric := 0;
  v_health numeric := 0;
  v_queue integer := 0;
BEGIN
  IF TG_OP = 'DELETE' THEN v_target_user := OLD.user_id;
  ELSE v_target_user := NEW.user_id; END IF;

  -- 1. REVENUE AT RISK: todo lo que no está OPTIMIZED.
  -- COALESCE porque NULL <> 'OPTIMIZED' evalua a NULL, no a true,
  -- y las filas con status nulo desaparecian del calculo.
  SELECT COALESCE(SUM(inventory_quantity * price), 0) INTO v_revenue
  FROM shopify_products
  WHERE user_id = v_target_user
    AND COALESCE(audit_status, '') <> 'OPTIMIZED';

  -- 2. CATALOG HEALTH
  SELECT COALESCE(AVG(audit_score), 0) INTO v_health
  FROM shopify_products
  WHERE user_id = v_target_user;

  -- 3. OPTIMIZATION QUEUE
  -- NEEDS_REVIEW queda fuera a proposito: es estado terminal, no cola.
  SELECT COUNT(id) INTO v_queue
  FROM shopify_products
  WHERE user_id = v_target_user
    AND audit_status IN ('PENDING_AUDIT', 'NEEDS_OPTIMIZATION', 'READY_TO_PUBLISH', 'ERROR');

  INSERT INTO user_kpis (user_id, revenue_at_risk, health_score_avg, items_in_queue, updated_at)
  VALUES (v_target_user, v_revenue, v_health, v_queue, now())
  ON CONFLICT (user_id) DO UPDATE SET
    revenue_at_risk = EXCLUDED.revenue_at_risk,
    health_score_avg = EXCLUDED.health_score_avg,
    items_in_queue = EXCLUDED.items_in_queue,
    updated_at = now();

  RETURN NULL;
END;
$function$;

-- B3: sin SECURITY DEFINER (invoker, para que RLS aplique).
CREATE OR REPLACE FUNCTION public.get_priority_products(p_limit integer DEFAULT 50)
 RETURNS TABLE(id bigint, shopify_id text, current_title text, audit_status text, audit_score integer, image_url text, inventory_quantity integer, price numeric, sales_last_7_days integer, created_at timestamp with time zone, priority_score numeric)
 LANGUAGE sql
 STABLE
 SET search_path TO 'public'
AS $function$
    select
        p.id::bigint,
        p.shopify_id::text,
        p.current_title::text,
        p.audit_status::text,
        p.audit_score::int,
        p.image_url::text,
        p.inventory_quantity::int,
        p.price::numeric,
        p.sales_last_7_days::int,
        p.created_at::timestamptz,
        (coalesce(p.inventory_quantity, 0) * coalesce(p.price, 0))
            * ((100 - least(greatest(coalesce(p.audit_score, 0), 0), 100))::numeric / 100)
          as priority_score
    from shopify_products p
    where p.user_id = auth.uid()
      and coalesce(p.audit_status, '') <> 'OPTIMIZED'
    order by priority_score desc, p.updated_at desc
    limit p_limit;
$function$;