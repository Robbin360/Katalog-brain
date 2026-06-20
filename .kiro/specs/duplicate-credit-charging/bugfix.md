# Bugfix Requirements Document

## Introduction

Este bugfix resuelve el **cobro duplicado de créditos** en el sistema Katalog-Brain, donde los usuarios son cobrados 2 veces por el mismo producto:
- Una vez en `save_to_supabase()` (línea ~1010) cuando `auto_pilot=false`
- Una segunda vez en `publish_to_shopify_node()` (línea ~1130) sin verificar si ya se cobró previamente

Adicionalmente, el sistema actual permite un escenario de explotación donde los usuarios pueden activar Auto-Pilot, optimizar 1000 productos sin cargo, desactivar Auto-Pilot y nunca publicar, resultando en 0 créditos cobrados.

La solución implementará un sistema **Reserve → Commit** con billing híbrido basado en el uso real de recursos:
- **1 crédito base** por optimización estándar
- **+1 crédito adicional** si se activa el `researcher_agent` (búsqueda web)
- **Cap máximo: 2 créditos por producto**

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN `auto_pilot=false` AND `save_to_supabase()` es llamado THEN el sistema cobra 1 crédito vía `charge_profile_credit()`

1.2 WHEN `publish_to_shopify_node()` es llamado THEN el sistema SIEMPRE cobra 1 crédito adicional vía `charge_profile_credit()` sin verificar cobros previos

1.3 WHEN un usuario activa Auto-Pilot, optimiza N productos, desactiva Auto-Pilot y nunca los publica THEN el sistema cobra 0 créditos (explotación del modelo de billing)

1.4 WHEN ocurre un early failure (nodos 0-2) THEN el sistema cobra créditos sin haber generado valor al usuario

1.5 WHEN `researcher_agent` es activado (búsqueda web costosa) THEN el sistema NO cobra el costo adicional de este recurso

### Expected Behavior (Correct)

2.1 WHEN el pipeline inicia en `start_processing` THEN el sistema SHALL reservar 1 crédito base sin cobrarlo

2.2 WHEN `orchestrator_node` detecta necesidad de investigación THEN el sistema SHALL reservar +1 crédito adicional (máximo 2 créditos totales)

2.3 WHEN `publish_to_shopify_node` publica exitosamente THEN el sistema SHALL hacer commit de los créditos reservados y marcar el producto como `OPTIMIZED`

2.4 WHEN ocurre un early failure (nodos 0-2) THEN el sistema SHALL reembolsar 100% de los créditos reservados

2.5 WHEN ocurre un late failure (nodo 3+) THEN el sistema SHALL hacer commit de créditos pero NO reembolsarlos

2.6 WHEN un producto falla 2+ veces en 24 horas THEN el sistema SHALL otorgar +1 crédito de compensación automática

2.7 WHEN `save_to_supabase()` guarda una propuesta THEN el sistema SHALL guardar sin cobrar (solo almacena, no publica)

### Unchanged Behavior (Regression Prevention)

3.1 WHEN el flujo completo se ejecuta exitosamente THEN el sistema SHALL CONTINUE TO marcar productos como `OPTIMIZED` en la base de datos

3.2 WHEN `save_to_supabase()` es llamado THEN el sistema SHALL CONTINUE TO guardar `ai_proposal` en formato JSONB

3.3 WHEN un producto ya tiene `audit_status = OPTIMIZED` THEN el sistema SHALL CONTINUE TO rechazar reprocesamiento para evitar amnesia de estado

3.4 WHEN se ejecuta el flujo completo THEN el sistema SHALL CONTINUE TO invocar los agentes en el orden correcto (orchestrator → researcher → writer → critic)

3.5 WHEN un usuario no tiene créditos suficientes THEN el sistema SHALL CONTINUE TO rechazar el procesamiento antes de iniciar el pipeline
