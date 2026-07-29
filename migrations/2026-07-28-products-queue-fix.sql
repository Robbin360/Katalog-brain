-- Migration: products_queue fix + security grants
-- FECHA: 2026-07-28
-- RIESGO: ALTO — modifica funciones SECURITY DEFINER y revoca permisos.
-- EJECUTAR SOLO EN STAGING PRIMERO.
-- NO EJECUTAR automáticamente. El propietario debe revisar y ejecutar.

-- ============================================================
-- 1. fix trigger_auto_audit — deduplicación + tenant filter
-- ============================================================
CREATE OR REPLACE FUNCTION public.trigger_auto_audit()
RETURNS TRIGGER
SECURITY DEFINER
SET search_path = public, pg_temp
LANGUAGE plpgsql AS $$
DECLARE
  v_exists BIGINT;
  v_source_version TEXT;
BEGIN
  -- Calcular source_version con hash SHA-256 del contenido fuente
  v_source_version := encode(
    sha256(
      COALESCE(NEW.current_title, '') ||
      COALESCE(NEW.current_body_html, '') ||
      COALESCE(NEW.vendor, '') ||
      COALESCE(NEW.tags, '') ||
      COALESCE(NEW.price::text, '')
    ),
    'hex'
  );

  -- Actualizar source_version en shopify_products
  NEW.source_version := v_source_version;

  -- Solo insertar en products_queue si:
  -- 1. El producto necesita auditoría (PENDING_AUDIT o NEEDS_OPTIMIZATION)
  -- 2. No existe ya un ticket duplicado para mismo user_id + shopify_product_db_id + source_version
  IF NEW.audit_status IN ('PENDING_AUDIT', 'NEEDS_OPTIMIZATION') THEN
    SELECT COUNT(*) INTO v_exists
    FROM public.products_queue
    WHERE user_id = NEW.user_id
      AND shopify_product_db_id = NEW.id
      AND source_version = v_source_version
      AND status != 'DONE';

    IF v_exists = 0 THEN
      INSERT INTO public.products_queue (
        user_id,
        shopify_product_db_id,
        source_version,
        status,
        created_at,
        updated_at
      ) VALUES (
        NEW.user_id,
        NEW.id,
        v_source_version,
        'PENDING',
        NOW(),
        NOW()
      );
    END IF;
  END IF;

  RETURN NEW;
END;
$$;

-- ============================================================
-- 2. fix update_audit_status — validar ownership + idempotencia
-- ============================================================
CREATE OR REPLACE FUNCTION public.update_audit_status(
  p_queue_id BIGINT,
  p_new_status TEXT,
  p_error_log TEXT DEFAULT NULL
)
RETURNS VOID
SECURITY DEFINER
SET search_path = public, pg_temp
LANGUAGE plpgsql AS $$
DECLARE
  v_queue RECORD;
  v_product RECORD;
BEGIN
  -- Leer queue row con lock preventivo
  SELECT * INTO v_queue
  FROM public.products_queue
  WHERE id = p_queue_id
  FOR UPDATE;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Queue row not found: %', p_queue_id;
  END IF;

  -- Si ya está DONE, salir (idempotente)
  IF v_queue.status = 'DONE' THEN
    RETURN;
  END IF;

  -- Validar que el producto existe y pertenece al mismo user_id
  SELECT * INTO v_product
  FROM public.shopify_products
  WHERE id = v_queue.shopify_product_db_id;

  IF NOT FOUND THEN
    RAISE EXCEPTION 'Product not found for queue row: %', p_queue_id;
  END IF;

  IF v_product.user_id != v_queue.user_id THEN
    RAISE EXCEPTION 'Ownership mismatch: product user_id % != queue user_id %',
      v_product.user_id, v_queue.user_id;
  END IF;

  -- Actualizar queue
  UPDATE public.products_queue
  SET status = p_new_status,
      error_log = COALESCE(p_error_log, error_log),
      updated_at = NOW(),
      completed_at = CASE WHEN p_new_status = 'DONE' THEN NOW() ELSE completed_at END
  WHERE id = p_queue_id;

  -- Actualizar producto solo si el estado es terminal
  IF p_new_status = 'DONE' THEN
    UPDATE public.shopify_products
    SET audit_status = 'OPTIMIZED',
        updated_at = NOW()
    WHERE id = v_queue.shopify_product_db_id;
  END IF;
END;
$$;

-- ============================================================
-- 3. Revocar EXECUTE en funciones de secretos a roles públicos
-- ============================================================
-- decrypt_shopify_token y encrypt_shopify_token deben ser backend-only.
REVOKE EXECUTE ON FUNCTION public.decrypt_shopify_token(TEXT) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.encrypt_shopify_token(TEXT) FROM PUBLIC, anon, authenticated;

-- ============================================================
-- 4. Revocar EXECUTE en funciones internas a roles públicos
-- ============================================================
REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.refresh_user_kpis(UUID) FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.trigger_auto_audit() FROM PUBLIC, anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.update_audit_status(BIGINT, TEXT, TEXT) FROM PUBLIC, anon, authenticated;

-- ============================================================
-- 5. Verificar increment_profile_credits_used antes de revocar
--    Buscar consumidores:
--    SELECT * FROM information_schema.routine_usage
--    WHERE routine_name = 'increment_profile_credits_used';
--    Si no tiene consumidores, revocar:
-- REVOKE EXECUTE ON FUNCTION public.increment_profile_credits_used(UUID, INT) FROM PUBLIC, anon, authenticated;
--    Si tiene consumidores, NO revocar hasta migrar a reserve/commit/refund.
-- ============================================================

-- ============================================================
-- 6. Corregir handle_new_user — usar inglés en defaults
-- ============================================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
SECURITY DEFINER
SET search_path = public, pg_temp
LANGUAGE plpgsql AS $$
BEGIN
  INSERT INTO public.user_profiles (id, plan_tier, created_at, updated_at)
  VALUES (
    NEW.id,
    'professional',
    NOW(),
    NOW()
  );
  -- Si existe la tabla profiles, también insertar
  INSERT INTO public.profiles (id, plan_tier, created_at, updated_at)
  VALUES (
    NEW.id,
    'professional',
    NOW(),
    NOW()
  )
  ON CONFLICT (id) DO NOTHING;

  RETURN NEW;
END;
$$;
