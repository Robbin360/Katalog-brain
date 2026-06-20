# Bugfix Requirements Document

## Introduction

Este documento especifica los requerimientos para corregir 4 vulnerabilidades críticas en el sistema de facturación de créditos de Katalog-Brain que causan pérdida de ingresos (~70% de revenue loss), costos no recuperados (~350% cost overruns), y permiten exploits por parte de usuarios.

**Impacto del Bug:**
- **Revenue Loss**: ~70% (usuarios evitando el sistema por cobros duplicados)
- **Cost Overruns**: ~350% (procesamiento sin cobro)
- **Margin Impact**: Reducción de 80% esperado a 59.6% actual

**Sistema Afectado:** Pipeline LangGraph de optimización de productos (`core/graph.py`, líneas 990-1180)

---

## Bug Analysis

### Current Behavior (Defect)

#### 1. Duplicate Credit Charging

1.1 WHEN un usuario en modo Manual aprueba un producto en estado `READY_TO_PUBLISH` y el Auto-Pilot está activado THEN el sistema cobra 1 crédito en `save_to_supabase` (línea 1010) y luego cobra 1 crédito adicional en `publish_to_shopify_node` (línea 1130), resultando en un cobro total de 2 créditos por 1 producto

1.2 WHEN un producto transita de Manual a Auto-Pilot después de optimización THEN el sistema cobra el crédito dos veces en lugar de una sola vez

1.3 WHEN el flujo ejecuta `route_after_save` con `auto_pilot_enabled=True` THEN el sistema no valida si ya cobró el crédito en el nodo anterior antes de publicar

#### 2. Auto-Pilot Deactivation Exploit

1.4 WHEN un usuario activa Auto-Pilot para procesar 1000 productos THEN el sistema marca los productos como `NEEDS_OPTIMIZATION` y el worker `auto_pilot_patrol` los procesa sin cobrar créditos en `save_to_supabase` (porque `auto_pilot_enabled=True` salta el cobro)

1.5 WHEN el usuario desactiva Auto-Pilot antes de que `publish_to_shopify_node` se ejecute THEN el sistema nunca cobra créditos, permitiendo al usuario procesar productos gratis

1.6 WHEN `save_to_supabase` verifica `if not state.get("auto_pilot_enabled", False)` THEN el sistema NO cobra créditos para productos de Auto-Pilot, creando una brecha de cobro si el usuario manipula el estado

#### 3. Credit Exhaustion Race Condition

1.7 WHEN el worker `auto_pilot_patrol` reserva un lote de productos basado en `credits_remaining` al inicio del ciclo THEN el sistema NO valida la disponibilidad de créditos antes de cada nodo del pipeline

1.8 WHEN múltiples productos se procesan en paralelo y los créditos se agotan durante la ejecución THEN el sistema completa el pipeline sin validar balance negativo, resultando en procesamiento no pagado

1.9 WHEN `charge_profile_credit` falla (error de red, timeout de Supabase) THEN el sistema NO revierte el procesamiento, permitiendo que el producto avance sin cobro

#### 4. No Transaction Tracking

1.10 WHEN el sistema cobra un crédito mediante `supabase.rpc("increment_profile_credits_used")` THEN no se registra un audit trail con timestamp, product_id, user_id, y tipo de operación

1.11 WHEN ocurre un error de facturación o un usuario reporta cobro incorrecto THEN no existe forma de auditar qué productos fueron cobrados, cuándo, ni por qué razón

1.12 WHEN se requiere emitir un refund THEN no hay tabla de transacciones que respalde la devolución con evidencia de la operación original

---

### Expected Behavior (Correct)

#### 1. Single Credit Charge per Product

2.1 WHEN un producto completa el pipeline de optimización en modo Manual o Auto-Pilot THEN el sistema SHALL cobrar exactamente 1 crédito base (optimización) + 1 crédito adicional (SOLO si `researcher_agent` fue activado), con un máximo de 2 créditos por producto

2.2 WHEN el sistema cobra créditos THEN el cobro SHALL ocurrir en un único punto del flujo (al inicio del pipeline mediante patrón Reserve-Commit), NO en múltiples nodos

2.3 WHEN un producto transita de `READY_TO_PUBLISH` a `OPTIMIZED` THEN el sistema SHALL validar que el crédito ya fue reservado y NO cobrará créditos adicionales

#### 2. Reserve-Commit Pattern with Refund Policy

2.4 WHEN un producto inicia el pipeline THEN el sistema SHALL reservar 1 crédito base inmediatamente en un estado "reserved" (NO cobrado aún)

2.5 WHEN el `orchestrator_node` decide activar `researcher_agent` THEN el sistema SHALL reservar 1 crédito adicional en estado "reserved"

2.6 WHEN el producto alcanza estado `OPTIMIZED` exitosamente THEN el sistema SHALL commit todas las reservas a "charged"

2.7 WHEN el pipeline falla en fases tempranas (fetch_data, orchestrator) THEN el sistema SHALL refund 100% de los créditos reservados

2.8 WHEN el pipeline falla en fases tardías (después de researcher/writer/critic) THEN el sistema SHALL refund 0% de los créditos reservados Y SHALL otorgar 1 crédito de compensación gratuito si el usuario tuvo 2+ fallos en las últimas 24 horas

#### 3. Credit Validation at Critical Checkpoints

2.9 WHEN el pipeline inicia en `start_processing` THEN el sistema SHALL validar que `credits_total - credits_used >= 1` antes de marcar el producto como `PROCESSING`

2.10 WHEN el `orchestrator_node` decide activar `researcher_agent` THEN el sistema SHALL validar que `credits_total - credits_used >= 2` antes de invocar al investigador

2.11 WHEN `charge_profile_credit` retorna `False` (error de cobro) THEN el sistema SHALL abortar el pipeline, marcar el producto como `ERROR`, y NO permitir que avance a nodos posteriores

2.12 WHEN el worker `auto_pilot_patrol` reserva un lote de productos THEN el sistema SHALL re-validar créditos disponibles para cada producto individual antes de procesar

#### 4. Transaction Audit Trail

2.13 WHEN el sistema reserva créditos THEN el sistema SHALL insertar un registro en la tabla `credit_transactions` con campos: `transaction_id` (UUID), `user_id`, `product_id`, `operation_type` ('reserve'), `credits_amount`, `status` ('reserved'), `timestamp`, `context` (JSON con metadata del pipeline)

2.14 WHEN el sistema cobra créditos THEN el sistema SHALL actualizar el registro existente cambiando `status` de 'reserved' a 'charged' y agregando `charged_at` timestamp

2.15 WHEN el sistema refund créditos THEN el sistema SHALL actualizar el registro cambiando `status` a 'refunded', decrementar `credits_used`, e insertar el motivo en campo `refund_reason`

2.16 WHEN un administrador consulta transacciones THEN el sistema SHALL permitir filtrar por `user_id`, `product_id`, `status`, y rango de fechas para auditoría completa

---

### Unchanged Behavior (Regression Prevention)

#### 1. Pipeline Flow Integrity

3.1 WHEN un producto en estado `STABLE_PERFORMING`, `MONITORING`, `BENCHMARK`, o `INVESTIGATE_CAUSE` es bloqueado por `do_not_harm_check` THEN el sistema SHALL CONTINUE TO omitir el cobro de créditos (comportamiento correcto actual)

3.2 WHEN un producto falla validación en `critic` después de 3 intentos THEN el sistema SHALL CONTINUE TO marcarlo como `NEEDS_OPTIMIZATION` sin cobrar créditos adicionales (correcto)

3.3 WHEN el worker `auto_pilot_patrol` detecta productos en estado `ERROR` con `retry_attempts < 3` y `next_retry_at <= now` THEN el sistema SHALL CONTINUE TO reintentarlos sin cobrar créditos adicionales por retry

#### 2. User Experience

3.4 WHEN un usuario en plan Free (0 créditos) intenta optimizar un producto THEN el sistema SHALL CONTINUE TO rechazar la operación con mensaje "Insufficient credits"

3.5 WHEN un usuario en plan Pro/Business tiene créditos disponibles THEN el sistema SHALL CONTINUE TO procesar productos automáticamente via `auto_pilot_patrol` en background

3.6 WHEN el Auto-Pilot worker procesa productos cada 30 segundos THEN el sistema SHALL CONTINUE TO respetar el límite de `MAX_PRODUCTS_PER_PATROL = 3` por ciclo

#### 3. Error Handling

3.7 WHEN un producto falla por error 429 (Rate Limit) THEN el sistema SHALL CONTINUE TO calcular `next_retry_at` y incrementar `retry_attempts` sin cobrar créditos adicionales

3.8 WHEN un producto falla por error 401/404 (error fatal) THEN el sistema SHALL CONTINUE TO marcar `retry_attempts = 3` para detener reintentos

3.9 WHEN el Zombie Sweeper detecta productos atascados en `PROCESSING` por >15 minutos THEN el sistema SHALL CONTINUE TO devolverlos a estado `ERROR` con `retry_attempts++`

#### 4. Database Schema Integrity

3.10 WHEN el sistema actualiza `shopify_products.audit_status` THEN el sistema SHALL CONTINUE TO usar los valores válidos: `PROCESSING`, `NEEDS_OPTIMIZATION`, `READY_TO_PUBLISH`, `OPTIMIZED`, `ERROR`, `STABLE_PERFORMING`, `MONITORING`, `BENCHMARK`, `INVESTIGATE_CAUSE`

3.11 WHEN el sistema actualiza `profiles.credits_used` THEN el sistema SHALL CONTINUE TO usar la función RPC `increment_profile_credits_used` para operaciones atómicas

3.12 WHEN el sistema registra optimizaciones en tabla `optimizations` THEN el sistema SHALL CONTINUE TO incluir todos los campos actuales: `user_id`, `product_id`, `title_generated`, `description_generated`, `title_previous`, `description_previous`, `framework_used`, `tone_used`, `description_length`, `status`

---

## Bug Condition Formalization

### Bug Condition Functions

#### C1: Duplicate Charging Condition
```pascal
FUNCTION isDuplicateChargingBug(state: KatalogState)
  INPUT: state of type KatalogState
  OUTPUT: boolean
  
  // Detecta cuando un producto será cobrado dos veces
  RETURN (
    state.auto_pilot_enabled = true AND
    state.current_status = "READY_TO_PUBLISH" AND
    hasProposal(state.final_proposal) = true
  ) OR (
    state.auto_pilot_enabled = false AND
    route_after_save(state) = "publish_to_shopify"
  )
END FUNCTION
```

#### C2: Auto-Pilot Exploit Condition
```pascal
FUNCTION isAutoPilotExploitBug(state: KatalogState)
  INPUT: state of type KatalogState
  OUTPUT: boolean
  
  // Detecta cuando un usuario puede evadir cobro desactivando Auto-Pilot
  RETURN (
    state.auto_pilot_enabled = true AND
    state.current_status = "NEEDS_OPTIMIZATION" AND
    creditChargedInSaveNode = false
  )
END FUNCTION
```

#### C3: Race Condition Exploit
```pascal
FUNCTION isRaceConditionBug(user: Profile, product: Product)
  INPUT: user of type Profile, product of type Product
  OUTPUT: boolean
  
  // Detecta cuando créditos se agotan durante procesamiento
  credits_at_start ← user.credits_total - user.credits_used
  credits_at_checkpoint ← getCurrentCredits(user.id)
  
  RETURN (
    credits_at_start >= 1 AND
    credits_at_checkpoint < 1 AND
    product.audit_status = "PROCESSING"
  )
END FUNCTION
```

#### C4: No Audit Trail Condition
```pascal
FUNCTION isAuditTrailMissing(user_id: UUID, product_id: BIGINT)
  INPUT: user_id of type UUID, product_id of type BIGINT
  OUTPUT: boolean
  
  // Detecta cuando un cobro ocurrió sin registro
  profile_credits_used_incremented ← wasIncrementCalled(user_id)
  transaction_exists ← existsInCreditTransactions(user_id, product_id)
  
  RETURN (
    profile_credits_used_incremented = true AND
    transaction_exists = false
  )
END FUNCTION
```

---

### Fix Validation Properties

#### Property 1: Single Charge Guarantee
```pascal
// FOR ALL productos que completan el pipeline exitosamente
FOR ALL product WHERE completedPipeline(product) = true DO
  credit_charges ← countCreditCharges(product.user_id, product.id)
  researcher_activated ← product.orchestrator_plan.activate_researcher_agent
  
  expected_charges ← 1 + (researcher_activated ? 1 : 0)
  
  ASSERT credit_charges = expected_charges
END FOR
```

#### Property 2: Reserve-Commit Correctness
```pascal
// FOR ALL productos en procesamiento
FOR ALL product WHERE product.audit_status = "PROCESSING" DO
  transactions ← getCreditTransactions(product.user_id, product.id)
  
  IF product.pipeline_completed = true THEN
    ASSERT ALL tx IN transactions HAVE tx.status = "charged"
  END IF
  
  IF product.pipeline_failed_early = true THEN
    ASSERT ALL tx IN transactions HAVE tx.status = "refunded"
  END IF
END FOR
```

#### Property 3: No Negative Balance
```pascal
// FOR ALL usuarios en todo momento
FOR ALL user IN profiles DO
  credits_used ← user.credits_used
  credits_total ← user.credits_total
  
  ASSERT credits_used <= credits_total
END FOR
```

#### Property 4: Complete Audit Trail
```pascal
// FOR ALL transacciones de créditos
FOR ALL transaction IN credit_transactions DO
  ASSERT transaction.user_id IS NOT NULL
  ASSERT transaction.product_id IS NOT NULL
  ASSERT transaction.operation_type IN ['reserve', 'charge', 'refund']
  ASSERT transaction.status IN ['reserved', 'charged', 'refunded']
  ASSERT transaction.timestamp IS NOT NULL
  
  IF transaction.status = "charged" THEN
    ASSERT transaction.charged_at IS NOT NULL
  END IF
  
  IF transaction.status = "refunded" THEN
    ASSERT transaction.refund_reason IS NOT NULL
  END IF
END FOR
```

---

## Preservation Goal

```pascal
// FOR ALL inputs que NO activan los bug conditions
FOR ALL state WHERE (
  NOT isDuplicateChargingBug(state) AND
  NOT isAutoPilotExploitBug(state) AND
  NOT isRaceConditionBug(state.user, state.product) AND
  NOT isAuditTrailMissing(state.user_id, state.product_id)
) DO
  // F = función original (código actual)
  // F' = función corregida (después del fix)
  
  ASSERT F(state) = F'(state)
END FOR
```

**Interpretación**: Para todos los flujos que NO activan las 4 vulnerabilidades, el sistema corregido debe comportarse idénticamente al sistema actual.

---

## Counterexamples (Concrete Bug Demonstrations)

### Counterexample 1: Duplicate Charging
```python
# Input State
state = {
    "product_id": 12345,
    "user_id": "uuid-abc-123",
    "auto_pilot_enabled": False,  # Modo Manual
    "current_status": "READY_TO_PUBLISH",
    "final_proposal": {"new_title": "...", "new_body_html": "..."}
}

# Execution Flow
# 1. save_to_supabase ejecuta → cobra 1 crédito (línea 1010)
# 2. route_after_save retorna "publish_to_shopify" (porque auto_pilot se activó después)
# 3. publish_to_shopify_node ejecuta → cobra 1 crédito ADICIONAL (línea 1130)

# Expected: 1 crédito cobrado
# Actual: 2 créditos cobrados ❌
```

### Counterexample 2: Auto-Pilot Exploit
```python
# User Actions
# 1. Usuario activa Auto-Pilot en perfil
# 2. Sistema procesa 1000 productos vía auto_pilot_patrol
# 3. Productos pasan por save_to_supabase → NO cobran (auto_pilot_enabled=True)
# 4. Usuario desactiva Auto-Pilot ANTES de que publish_to_shopify_node se ejecute
# 5. Productos quedan en READY_TO_PUBLISH sin cobro

# Expected: 1000 créditos cobrados
# Actual: 0 créditos cobrados ❌
```

### Counterexample 3: Race Condition
```python
# Initial State
user.credits_total = 100
user.credits_used = 99  # 1 crédito disponible

# Worker Execution
auto_pilot_patrol reserva 3 productos (MAX_PRODUCTS_PER_PATROL)
# Producto 1 procesa → cobra 1 crédito → credits_used = 100
# Producto 2 intenta procesar → balance negativo → credits_used = 101 ❌
# Producto 3 intenta procesar → credits_used = 102 ❌

# Expected: Producto 1 procesa, Productos 2-3 rechazan
# Actual: 3 productos procesan con balance negativo ❌
```

### Counterexample 4: No Audit Trail
```sql
-- Query para validar audit trail
SELECT * FROM credit_transactions 
WHERE user_id = 'uuid-abc-123' AND product_id = 12345;

-- Expected: 1 registro con operation_type='charge', status='charged'
-- Actual: 0 registros (tabla no existe) ❌
```
