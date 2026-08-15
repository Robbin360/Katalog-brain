"""
core/quality_gate.py

Puerta de dos capas:

  1. DETERMINISTA (gratis, sin tokens): forma. Longitud de título,
     palabras mínimas, palabras prohibidas, stuffing, estructura.
     Además compara candidato contra actual en métricas objetivas, para
     no publicar un texto que empeora lo que ya existe.

  2. INSPECTOR (1 request): relevancia y corrección del lenguaje.
     Solo se ejecuta si la capa 1 aprobó. No tiene sentido gastar cuota
     revisando el español de un título de 9 caracteres.

Diseño anterior y por qué se retiró: el gate puntuaba texto viejo y nuevo
con una escala 0-100 y aprobaba si la diferencia era >= 5. Costaba 2
requests, y a temperatura 0 el modelo solo emitía múltiplos de 5, así que
el margen mínimo estaba dentro del ruido. Medido contra el catálogo real,
los puntajes además estaban invertidos.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.inspector_agent import RelevanceVerdict, check_relevance
from core.deterministic_score import (
    DeterministicScoreResult,
    evaluate_deterministic,
    quality_percent,
)


@dataclass(frozen=True)
class GateVerdict:
    deterministic: DeterministicScoreResult
    relevance: RelevanceVerdict | None
    regressions: list[str]

    @property
    def passed(self) -> bool:
        if not self.deterministic.passes_gate:
            return False
        if self.regressions:
            return False
        # relevance es None solo si la capa 1 rechazó antes de gastar cuota.
        return self.relevance is not None and self.relevance.passed

    @property
    def audit_score(self) -> int:
        """
        Valor para la columna shopify_products.audit_score. Determinista:
        derivado de checks verificables, no de la opinión de un modelo.
        """
        return quality_percent(self.deterministic)

    @property
    def blocking_reasons(self) -> list[str]:
        reasons = list(self.deterministic.failures)
        reasons.extend(self.regressions)
        if self.relevance is not None and not self.relevance.passed:
            reasons.append(self.relevance.as_log_line())
        return reasons

    def as_log_line(self) -> str:
        estado = "APROBADO" if self.passed else "RECHAZADO"
        if self.passed:
            return (
                f"Gate {estado} (score {self.audit_score}): forma correcta, "
                "producto correcto, lenguaje correcto."
            )
        return f"Gate {estado}: " + " | ".join(self.blocking_reasons)

    def as_dict(self) -> dict:
        return {
            "passed": self.passed,
            "audit_score": self.audit_score,
            "blocking_reasons": self.blocking_reasons,
            "deterministic_failures": list(self.deterministic.failures),
            "regressions": list(self.regressions),
            "inflesz_band": self.deterministic.inflesz_band,
            "concrete_facts": self.deterministic.concrete_facts_count,
            "title_length": self.deterministic.title_length,
            "relevance_evaluated": self.relevance is not None,
            "same_product": self.relevance.same_product if self.relevance else None,
            "language_ok": self.relevance.language_ok if self.relevance else None,
            "relevance_evidence": self.relevance.evidence if self.relevance else None,
        }


def detect_regressions(
    current: DeterministicScoreResult,
    candidate: DeterministicScoreResult,
) -> list[str]:
    """
    Comparación determinista, sin tokens. Bloquea propuestas que degradan
    lo publicado en dimensiones objetivas.

    Motivado por el caso real del producto 1009: el writer lo llevó de
    "Tabla All-Mountain de Alto Rendimiento | Control y Velocidad Total"
    a "Tabla Pro Extreme" en cuatro publicaciones sucesivas, y el gate
    anterior aprobó cada paso.
    """
    regressions: list[str] = []

    if candidate.title_length < current.title_length * 0.6:
        regressions.append(
            f"Regresion: el titulo se acorta de {current.title_length} a "
            f"{candidate.title_length} caracteres."
        )

    if candidate.body_word_count < current.body_word_count * 0.6:
        regressions.append(
            f"Regresion: la descripcion se acorta de {current.body_word_count} "
            f"a {candidate.body_word_count} palabras."
        )

    if current.concrete_facts_count > 0 and candidate.concrete_facts_count == 0:
        regressions.append(
            f"Regresion: se pierden los {current.concrete_facts_count} datos "
            "concretos (medidas o cifras) que tenia el texto publicado."
        )

    if current.has_structural_elements and not candidate.has_structural_elements:
        regressions.append(
            "Regresion: se pierde la estructura (listas o encabezados) y queda "
            "un muro de texto."
        )

    return regressions


async def evaluate_rewrite(
    *,
    current_title: str | None,
    current_body_html: str | None,
    candidate_title: str | None,
    candidate_body_html: str | None,
    vendor: str | None = None,
    tags: str | None = None,
    forbidden_words: list[str] | None = None,
) -> GateVerdict:
    """
    Evalúa una propuesta. Consume 1 request del modelo, y solo si la capa
    determinista aprobó primero.
    """
    candidate_result = evaluate_deterministic(
        candidate_title, candidate_body_html, forbidden_words
    )
    current_result = evaluate_deterministic(
        current_title, current_body_html, forbidden_words
    )
    regressions = detect_regressions(current_result, candidate_result)

    # Fallo temprano: no gastamos cuota en un texto ya descartado por forma.
    if not candidate_result.passes_gate or regressions:
        return GateVerdict(
            deterministic=candidate_result,
            relevance=None,
            regressions=regressions,
        )

    verdict = await check_relevance(
        reference_title=current_title,
        reference_vendor=vendor,
        reference_tags=tags,
        candidate_title=candidate_title,
        candidate_body_html=candidate_body_html,
    )

    return GateVerdict(
        deterministic=candidate_result,
        relevance=verdict,
        regressions=regressions,
    )