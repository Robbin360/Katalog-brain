"""
Tests del contrato SEO del writer: AIProposalOutput con seo_title y
seo_description, y pre_critic_checks del juez. Cero IA, cero red.

Regla dura del proyecto: si un test falla, se arregla el codigo, NUNCA
se debilita o borra el test para que pase.
"""

import pytest
from pydantic import ValidationError

from core.schemas import AIProposalOutput
from agents.critic_agent import pre_critic_checks


TITLE_40 = "T" * 40
BODY = "<ul><li>Beneficio concreto del producto para el comprador: material resistente y durabilidad probada.</li></ul>"
SEO_TITLE_40 = "S" * 40
SEO_DESC_110 = "D" * 110


def _valid_proposal(overrides: dict | None = None) -> dict:
    proposal = {
        "new_title": TITLE_40,
        "new_body_html": BODY,
        "seo_title": SEO_TITLE_40,
        "seo_description": SEO_DESC_110,
        "audit_log": ["Titulo optimizado."],
    }
    if overrides:
        proposal.update(overrides)
    return proposal


# ---------------------------------------------------------------------
# Contrato del schema: las longitudes viven en el schema (output_retries
# corrige dentro del ciclo de escritura), no en pre_critic_checks.
# ---------------------------------------------------------------------

def test_schema_rechaza_seo_description_corta():
    with pytest.raises(ValidationError):
        AIProposalOutput(**_valid_proposal({"seo_description": "D" * 90}))


def test_schema_rechaza_seo_description_larga():
    with pytest.raises(ValidationError):
        AIProposalOutput(**_valid_proposal({"seo_description": "D" * 200}))


def test_schema_rechaza_seo_title_corto():
    with pytest.raises(ValidationError):
        AIProposalOutput(**_valid_proposal({"seo_title": "S" * 30}))


def test_schema_valido_con_cinco_campos_construye():
    proposal = AIProposalOutput(**_valid_proposal())
    assert proposal.seo_title == SEO_TITLE_40
    assert proposal.seo_description == SEO_DESC_110
    assert proposal.seo_title != proposal.new_title
    assert proposal.audit_log == ["Titulo optimizado."]


def test_schema_rechaza_seo_title_identico_a_new_title():
    """Regresion 2026-08-18: el escritor emitio seo_title identico a
    new_title y el Juez lo aprobo. El validador debe rechazarlo para que
    output_retries lo corrija dentro del ciclo de escritura."""
    with pytest.raises(ValidationError, match="seo_title must not be identical"):
        AIProposalOutput(**_valid_proposal({"seo_title": TITLE_40}))


def test_schema_acepta_seo_title_distinto_a_new_title():
    proposal = AIProposalOutput(**_valid_proposal())
    assert proposal.seo_title != proposal.new_title


# ---------------------------------------------------------------------
# pre_critic_checks: el haystack cubre todo el texto que se publica,
# incluidos los campos SEO nuevos.
# ---------------------------------------------------------------------

def test_pre_critic_detecta_prohibida_solo_en_seo_description():
    proposal = _valid_proposal({"seo_description": f"{SEO_DESC_110} prohibida"})
    rules = {"forbidden_words": ["prohibida"]}

    with pytest.raises(ValueError, match="Forbidden words found"):
        pre_critic_checks(proposal, rules)


def test_pre_critic_detecta_prohibida_solo_en_seo_title():
    proposal = _valid_proposal({"seo_title": f"prohibida {SEO_TITLE_40}"})
    rules = {"forbidden_words": ["prohibida"]}

    with pytest.raises(ValueError, match="Forbidden words found"):
        pre_critic_checks(proposal, rules)


def test_pre_critic_sin_palabra_prohibida_no_lanza():
    proposal = _valid_proposal()
    rules = {"forbidden_words": ["prohibida"]}

    pre_critic_checks(proposal, rules)