"""
scripts/audit_catalog_score.py

Error analysis del catálogo con el scorer determinista.
SOLO LECTURA: no escribe en Supabase, no llama a ninguna IA, no gasta cuota.

Objetivo: ver qué productos pasarían la puerta determinista y por qué
fallan los demás, ANTES de conectar el scorer al quality gate.

Uso:
    cd C:\\proyectos\\Katalog-brain
    .\\.venv\\Scripts\\activate
    python -m scripts.audit_catalog_score
"""

from __future__ import annotations

import os
import sys
from collections import Counter

from dotenv import load_dotenv
from supabase import create_client

from core.deterministic_score import (
    evaluate_deterministic,
    BODY_MIN_WORDS,
    TITLE_MIN_LENGTH,
    TITLE_MAX_LENGTH,
)

load_dotenv()

# Los nombres de variables varían entre proyectos: probamos los habituales
# y reportamos cuál se usó, en lugar de fallar con un KeyError opaco.
URL_CANDIDATES = ("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL", "SUPABASE_PROJECT_URL")
KEY_CANDIDATES = (
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_SERVICE_KEY",
    "SUPABASE_KEY",
    "SUPABASE_ANON_KEY",
)


def _first_env(names: tuple[str, ...]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value:
            return name, value
    return None, None


def main() -> int:
    url_name, url = _first_env(URL_CANDIDATES)
    key_name, key = _first_env(KEY_CANDIDATES)

    if not url or not key:
        print("No encontre credenciales de Supabase en .env")
        print(f"  Busque URL en: {', '.join(URL_CANDIDATES)}")
        print(f"  Busque KEY en: {', '.join(KEY_CANDIDATES)}")
        return 1

    print(f"Conectando con {url_name} + {key_name}\n")
    client = create_client(url, key)

    resp = (
        client.table("shopify_products")
        .select("id, audit_status, audit_score, current_title, current_body_html, ai_proposal")
        .order("id")
        .execute()
    )
    products = resp.data or []

    if not products:
        print("Sin productos visibles. Si usaste la anon key, RLS bloquea la lectura:")
        print("necesitas la service role key para leer todo el catalogo.")
        return 1

    print(f"Umbrales actuales: titulo {TITLE_MIN_LENGTH}-{TITLE_MAX_LENGTH} chars, "
          f"cuerpo minimo {BODY_MIN_WORDS} palabras\n")
    print("=" * 100)
    print("ESTADO ACTUAL PUBLICADO (current_title / current_body_html)")
    print("=" * 100)

    reasons: Counter[str] = Counter()
    passed = 0
    scores_ia: list[int] = []

    for p in products:
        result = evaluate_deterministic(p.get("current_title"), p.get("current_body_html"))
        mark = "PASA " if result.passes_gate else "FALLA"
        if result.passes_gate:
            passed += 1

        ia_score = p.get("audit_score")
        if isinstance(ia_score, int):
            scores_ia.append(ia_score)

        title = (p.get("current_title") or "")[:52]
        print(
            f"\n[{mark}] id={p['id']:<5} ia_score={ia_score if ia_score is not None else '-':<4} "
            f"{p.get('audit_status', '?')}"
        )
        print(f"        titulo ({result.title_length} chars): {title}")
        print(
            f"        palabras={result.body_word_count:<4} "
            f"hechos={result.concrete_facts_count:<3} "
            f"estructura={'si' if result.has_structural_elements else 'NO':<3} "
            f"inflesz={result.inflesz_band}"
        )
        if result.title_truncation_risk:
            print(f"        aviso: ~{result.title_pixel_width_estimate}px, riesgo de truncar en Google")
        for failure in result.failures:
            print(f"        - {failure}")
            reasons[failure.split("(")[0].strip().rstrip(":")] += 1

    total = len(products)
    print("\n" + "=" * 100)
    print(f"RESUMEN: {passed}/{total} pasan la puerta determinista "
          f"({passed * 100 // total if total else 0}%)")
    if scores_ia:
        print(f"Promedio de audit_score que dio la IA: {sum(scores_ia) / len(scores_ia):.1f}")
    print("\nMotivos de fallo, por frecuencia:")
    for reason, count in reasons.most_common():
        print(f"  {count:>3}x  {reason}")

    # Compara: la propuesta pendiente sería mejor o peor que lo publicado.
    print("\n" + "=" * 100)
    print("PROPUESTAS PENDIENTES (ai_proposal) vs LO PUBLICADO")
    print("=" * 100)

    compared = 0
    for p in products:
        proposal = p.get("ai_proposal") or {}
        if not isinstance(proposal, dict):
            continue
        new_title = proposal.get("new_title")
        new_body = proposal.get("new_body_html")
        if not new_title or not new_body:
            continue

        current = evaluate_deterministic(p.get("current_title"), p.get("current_body_html"))
        candidate = evaluate_deterministic(new_title, new_body)

        if new_title == (p.get("current_title") or ""):
            verdict = "IDENTICA a lo publicado"
        elif candidate.passes_gate and not current.passes_gate:
            verdict = "MEJORA: la propuesta pasa y lo actual no"
        elif not candidate.passes_gate and current.passes_gate:
            verdict = "EMPEORA: lo actual pasa y la propuesta no"
        elif candidate.passes_gate and current.passes_gate:
            verdict = "ambas pasan"
        else:
            verdict = "ambas fallan"

        compared += 1
        print(f"\nid={p['id']}  {verdict}")
        print(f"  actual   : {(p.get('current_title') or '')[:60]}")
        print(f"  propuesta: {new_title[:60]}")
        if not candidate.passes_gate:
            for failure in candidate.failures:
                print(f"    - {failure}")

    print(f"\n{compared} productos con propuesta comparable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())