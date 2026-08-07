from __future__ import annotations

from dataclasses import dataclass

from agents.inspector_agent import score_content

PUBLISH_MARGIN = 5


@dataclass(frozen=True)
class GateVerdict:
    old_score: int
    new_score: int
    old_reason: str
    new_reason: str
    margin: int = PUBLISH_MARGIN

    @property
    def delta(self) -> int:
        return self.new_score - self.old_score

    @property
    def passed(self) -> bool:
        return self.delta >= self.margin

    def as_log_line(self) -> str:
        estado = "APROBADO" if self.passed else "RECHAZADO"
        return (
            f"Gate {estado}: {self.old_score} -> {self.new_score} "
            f"({self.delta:+d}, minimo {self.margin}). Nuevo texto: {self.new_reason}"
        )

    def as_dict(self) -> dict:
        return {
            "old_score": self.old_score,
            "new_score": self.new_score,
            "delta": self.delta,
            "margin": self.margin,
            "passed": self.passed,
            "old_reason": self.old_reason,
            "new_reason": self.new_reason,
        }


async def evaluate_rewrite(
    *,
    old_title: str | None,
    old_body_html: str | None,
    new_title: str | None,
    new_body_html: str | None,
    margin: int = PUBLISH_MARGIN,
) -> GateVerdict:
    """Puntúa ambas versiones con el mismo juez, en la misma corrida."""
    old_score, old_reason = await score_content(old_title, old_body_html)
    new_score, new_reason = await score_content(new_title, new_body_html)
    return GateVerdict(
        old_score=old_score,
        new_score=new_score,
        old_reason=old_reason,
        new_reason=new_reason,
        margin=margin,
    )
