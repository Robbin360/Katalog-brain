"""
Tests de helpers deterministas: clasificacion de producto y extraccion de
hechos verificados desde metafields. Cero IA, cero red: corren en
milisegundos y no requieren mocks de Supabase ni de Gemini.
"""

import pytest

from core.helpers import classify_product_type, extract_verified_facts


# ---------------------------------------------------------------------
# extract_verified_facts: hechos NIVEL 1 emitidos por Shopify
# ---------------------------------------------------------------------

def test_extract_verified_facts_dimension_centimeters():
    metafields = {
        "test_data.snowboard_length": {
            "value": '{"value":159.0,"unit":"CENTIMETERS"}',
            "type": "dimension",
        }
    }
    assert extract_verified_facts(metafields) == ["Snowboard length: 159 cm"]


def test_extract_verified_facts_weight_pounds():
    metafields = {
        "test_data.snowboard_weight": {
            "value": '{"value":8.0,"unit":"POUNDS"}',
            "type": "weight",
        }
    }
    assert extract_verified_facts(metafields) == ["Snowboard weight: 8 lb"]


def test_extract_verified_facts_ignora_texto_libre():
    """test_data.binding_mount = "Optimistic" es basura generada: no debe
    producir hechos afirmables (verificado 2026-08-18)."""
    metafields = {
        "test_data.binding_mount": {
            "value": "Optimistic",
            "type": "single_line_text_field",
        }
    }
    assert extract_verified_facts(metafields) == []


def test_extract_verified_facts_none_o_vacio():
    assert extract_verified_facts(None) == []
    assert extract_verified_facts({}) == []


def test_extract_verified_facts_json_malformado_no_lanza():
    metafields = {
        "test_data.broken": {"value": '{"value":159.0', "type": "dimension"}
    }
    result = extract_verified_facts(metafields)
    assert isinstance(result, list)


# ---------------------------------------------------------------------
# classify_product_type: filas de shopify_products vs payload de Shopify
# ---------------------------------------------------------------------

def test_classify_product_type_tags_csv_detecta_artisanal():
    """save_products_to_db guarda tags como string CSV, no como lista."""
    product = {
        "product_type": "tabla",
        "tags": "Snowboard, Hecho a mano, Lote 12",
        "barcode": "",
        "sku": "",
    }
    assert classify_product_type(product) == "ARTISANAL"


def test_classify_product_type_acepta_product_type_y_productType():
    """La columna es product_type (snake_case); el payload crudo trae
    productType. Ambas deben clasificar igual."""
    row = {
        "product_type": "artesanal",
        "tags": [],
        "barcode": "",
        "sku": "",
    }
    raw_payload = {
        "productType": "artesanal",
        "tags": [],
        "barcode": "",
        "sku": "",
    }
    assert classify_product_type(row) == "ARTISANAL"
    assert classify_product_type(raw_payload) == "ARTISANAL"


# ---------------------------------------------------------------------
# classify_product_type: NON_PHYSICAL (categorías sin especificaciones)
# ---------------------------------------------------------------------

def test_classify_product_type_giftcard_no_fisico():
    product = {"product_type": "giftcard", "tags": [], "barcode": "", "sku": ""}
    assert classify_product_type(product) == "NON_PHYSICAL"


def test_classify_product_type_gift_card_mayusculas_con_espacio():
    product = {"product_type": "Gift Card", "tags": [], "barcode": "", "sku": ""}
    assert classify_product_type(product) == "NON_PHYSICAL"


def test_classify_product_type_snowboard_generic():
    product = {"product_type": "snowboard", "tags": [], "barcode": "", "sku": ""}
    assert classify_product_type(product) == "GENERIC"


def test_classify_product_type_barcode_manda_sobre_giftcard():
    """barcode es identidad, no categoria: manda sobre NON_PHYSICAL."""
    product = {"product_type": "giftcard", "tags": [], "barcode": "0072358", "sku": ""}
    assert classify_product_type(product) == "MANUFACTURED"