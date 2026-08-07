-- =============================================================
-- Migration: audit_score_integrity
-- Fecha: 2026-08-06
-- Descripción:
--   B1 - Trigger check_product_health (column-level, circuit-breaker)
--   B2 - RPC refresh_user_kpis (COALESCE seguro contra NULL)
--   B3 - RPC get_priority_products (priority_score por valor de inventario)
--
-- Aplicar en: SQL Editor de Supabase (Dashboard → SQL Editor → New query)
-- =============================================================


-- =============================================================
-- B1: Trigger function + trigger
-- =============================================================

CREATE OR REPLACE FUNCTION public.fn_check_product_health()
  RETURNS TRIGGER
  LANGUAGE plpgsql
  SECURITY DEFINER
AS $$
BEGIN
  -- Solo actuar si cambió alguna columna relevante para la calidad
  IF (
    NEW.sales_last_7_days IS DISTINCT FROM OLD.sales_last_7_days OR
    NEW.inventory_quantity IS DISTINCT FROM OLD.inventory_quantity OR
    NEW.current_title      IS DISTINCT FROM OLD.current_title      OR
    NEW.current_body_html  IS DISTINCT FROM OLD.current_body_html
  ) THEN
    -- Circuit-breaker: no re-encolar si ya está optimizado o falló 3+ veces
    IF (
      coalesce(NEW.audit_status, '') NOT IN ('OPTIMIZED', 'READY_TO_PUBLISH') AND
      coalesce(NEW.consecutive_failures, 0) < 3
    ) THEN
      NEW.audit_status := 'NEEDS_OPTIMIZATION';
    END IF;
  END IF;

  RETURN NEW;
END;
$$;

-- Eliminar el trigger previo si existe con distinta definición
DROP TRIGGER IF EXISTS check_product_health ON public.shopify_products;

CREATE TRIGGER check_product_health
  BEFORE UPDATE ON public.shopify_products
  FOR EACH ROW
  EXECUTE FUNCTION public.fn_check_product_health();


-- =============================================================
-- B2: refresh_user_kpis
-- =============================================================

CREATE OR REPLACE FUNCTION public.refresh_user_kpis(p_user_id uuid)
  RETURNS void
  LANGUAGE plpgsql
  SECURITY DEFINER
AS $$
BEGIN
  INSERT INTO public.user_kpis (user_id, revenue_at_risk, health_score_avg, items_in_queue, refreshed_at)
  SELECT
    p_user_id,
    -- Revenue at risk: valor del inventario de productos NO optimizados
    COALESCE(SUM(
      CASE
        WHEN COALESCE(p.audit_status, '') <> 'OPTIMIZED'
        THEN COALESCE(p.inventory_quantity, 0) * COALESCE(p.price, 0)
        ELSE 0
      END
    ), 0),
    -- Promedio de audit_score (0-100)
    COALESCE(AVG(p.audit_score), 0),
    -- Ítems pendientes de optimización
    COUNT(*) FILTER (
      WHERE COALESCE(p.audit_status, '') NOT IN ('OPTIMIZED', 'READY_TO_PUBLISH')
    )
  FROM public.shopify_products p
  WHERE p.user_id = p_user_id
  ON CONFLICT (user_id) DO UPDATE SET
    revenue_at_risk  = EXCLUDED.revenue_at_risk,
    health_score_avg = EXCLUDED.health_score_avg,
    items_in_queue   = EXCLUDED.items_in_queue,
    refreshed_at     = now();
END;
$$;


-- =============================================================
-- B3: RPC get_priority_products
-- priority_score = valor_inventario × (déficit_de_calidad / 100)
-- =============================================================

CREATE OR REPLACE FUNCTION public.get_priority_products(p_limit int DEFAULT 50)
  RETURNS TABLE (
    id                   uuid,
    user_id              uuid,
    shopify_product_id   text,
    current_title        text,
    audit_score          int,
    audit_status         text,
    last_audit_at        timestamptz,
    consecutive_failures int,
    inventory_quantity   int,
    price                numeric,
    priority_score       numeric
  )
  LANGUAGE sql
  SECURITY DEFINER
  STABLE
AS $$
  SELECT
    p.id,
    p.user_id,
    p.shopify_product_id,
    p.current_title,
    p.audit_score,
    p.audit_status,
    p.last_audit_at,
    p.consecutive_failures,
    p.inventory_quantity,
    p.price,
    -- Cuánto valor USD está en riesgo por baja calidad del listing
    (
      COALESCE(p.inventory_quantity, 0) * COALESCE(p.price, 0)
    ) * (
      (100 - LEAST(GREATEST(COALESCE(p.audit_score, 0), 0), 100))::numeric / 100
    ) AS priority_score
  FROM public.shopify_products p
  WHERE p.user_id = auth.uid()
  ORDER BY priority_score DESC NULLS LAST
  LIMIT p_limit;
$$;
