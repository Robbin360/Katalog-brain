"""
Tests del scorer determinista. Cero IA, cero red: deben correr en
milisegundos y no requieren mocks de Supabase ni de Gemini.

Regla dura del proyecto: si un test falla, se arregla el codigo, NUNCA
se debilita o borra el test para que pase.

Los dos ultimos tests son regresiones de la calibracion del 2026-08-15,
hecha midiendo el catalogo real de 18 productos.
"""

import pytest

from core.deterministic_score import (
    count_syllables_word_es,
    count_words_es,
    count_sentences_es,
    flesch_szigriszt_inflesz,
    inflesz_band,
    count_concrete_facts,
    has_structural_elements,
    find_forbidden_words,
    keyword_stuffing_ratio,
    estimate_title_pixel_width,
    strip_html_to_text,
    evaluate_deterministic,
    TITLE_MIN_LENGTH,
    TITLE_MAX_LENGTH,
    MAX_STUFFING_RATIO,
)


# ---------------------------------------------------------------------
# Silabas: casos verificados contra reglas RAE de diptongo/hiato
# ---------------------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    ("dia", 1),        # SIN tilde: "ia" es diptongo -> 1 silaba
    ("día", 2),        # CON tilde: la i rompe el diptongo -> hiato
    ("aire", 2),       # diptongo "ai" + "re"
    ("cae", 2),        # hiato: a+e ambas fuertes
    ("cielo", 2),      # diptongo "ie" + "lo"
    ("pais", 1),       # SIN tilde: "ai" es diptongo -> 1 silaba
    ("país", 2),       # CON tilde: hiato
    ("bueno", 2),      # diptongo "ue" + "no"
    ("maria", 2),      # SIN tilde: "ia" final es diptongo -> ma+ria
    ("María", 3),      # CON tilde: Ma+ri+a
    ("tabla", 2),
    ("snowboard", 3),  # aproximacion razonable para prestamo del ingles
    ("resistencia", 4),
])
def test_count_syllables_word_es(word, expected):
    assert count_syllables_word_es(word) == expected


def test_count_syllables_word_es_limitacion_conocida_sin_tilde():
    """
    Limitacion documentada, no oculta: el algoritmo detecta hiato SOLO
    por la presencia de tilde en la vocal debil, que es la regla
    ortografica del espanol. El scorer opera sobre el texto tal como se
    publica, no sobre la intencion fonetica del hablante.
    """
    assert count_syllables_word_es("maria") != count_syllables_word_es("María")


def test_count_syllables_word_es_sin_vocales():
    assert count_syllables_word_es("xyz") == 0


# ---------------------------------------------------------------------
# HTML -> texto y conteo de palabras/frases
# ---------------------------------------------------------------------

def test_strip_html_inserta_puntos_en_bloques():
    html = "<ul><li>Uno</li><li>Dos</li><li>Tres</li></ul>"
    text = strip_html_to_text(html)
    # Debe verse como 3 frases, no como una sola corrida.
    assert count_sentences_es(text) == 3


def test_strip_html_vacio():
    assert strip_html_to_text(None) == ""
    assert strip_html_to_text("") == ""


def test_count_words_es_ignora_numeros_sueltos():
    assert count_words_es("Tiene 5 ruedas y 2 frenos") == 4


def test_count_sentences_es_nunca_es_cero():
    assert count_sentences_es("sin puntuacion alguna") == 1


# ---------------------------------------------------------------------
# INFLESZ
# ---------------------------------------------------------------------

def test_inflesz_none_si_texto_muy_corto():
    assert flesch_szigriszt_inflesz("Tabla Pro") is None
    assert inflesz_band(None) == "indeterminado"


def test_inflesz_bandas_frontera():
    assert inflesz_band(39.9) == "muy_dificil"
    assert inflesz_band(40.0) == "algo_dificil"
    assert inflesz_band(54.9) == "algo_dificil"
    assert inflesz_band(55.0) == "normal"
    assert inflesz_band(64.9) == "normal"
    assert inflesz_band(65.0) == "bastante_facil"
    assert inflesz_band(79.9) == "bastante_facil"
    assert inflesz_band(80.0) == "muy_facil"


def test_inflesz_texto_normal_cae_en_banda_razonable():
    text = (
        "Esta tabla es resistente. Soporta el frío intenso. "
        "Tiene un borde firme. Se controla con facilidad en la nieve. "
        "Es ideal para principiantes y expertos."
    )
    score = flesch_szigriszt_inflesz(text)
    assert score is not None
    assert inflesz_band(score) in {"normal", "bastante_facil", "muy_facil"}


# ---------------------------------------------------------------------
# Hechos concretos y estructura
# ---------------------------------------------------------------------

def test_concrete_facts_detecta_numero_mas_unidad():
    text = "Mide 158 cm de largo y pesa 3.2 kg, soporta hasta 120 kg."
    assert count_concrete_facts(text) == 3


def test_concrete_facts_cero_en_texto_generico():
    assert count_concrete_facts("Es una tabla muy buena y resistente") == 0


def test_has_structural_elements_con_lista():
    assert has_structural_elements("<ul><li>Uno</li><li>Dos</li></ul>") is True


def test_has_structural_elements_muro_de_texto():
    assert has_structural_elements("<p>Solo un parrafo sin nada mas aqui.</p>") is False


# ---------------------------------------------------------------------
# Palabras prohibidas y stuffing
# ---------------------------------------------------------------------

def test_find_forbidden_words_case_insensitive():
    found = find_forbidden_words("Este es el MEJOR producto del mundo", ["mejor"])
    assert found == ["mejor"]


def test_find_forbidden_words_ninguna():
    assert find_forbidden_words("Texto limpio", ["prohibida"]) == []


def test_keyword_stuffing_detecta_repeticion_excesiva():
    text = " ".join(["snowboard"] * 20 + ["tabla", "resistente", "control", "nieve"])
    word, ratio = keyword_stuffing_ratio(text)
    assert word == "snowboard"
    assert ratio > 0.5


def test_stuffing_tolera_palabra_de_categoria():
    """
    Regresion 2026-08-15: 'snowboard' repetido en un producto de snowboard
    es la palabra de la categoria, no stuffing. Con el umbral anterior
    (8%) se marcaba como falso positivo en el catalogo real (1012, 1013,
    1014).

    Los rellenos son variados a proposito. En la primera version de este
    test se repetian pocas palabras muchas veces, asi que la palabra mas
    frecuente resultaba ser un relleno (22.7%) y no 'snowboard': el test
    medía otra cosa de la que decía medir.
    """
    fillers = [
        "control", "nieve", "borde", "descenso", "flexible",
        "resistente", "montana", "pendiente", "agarre", "estable",
        "ligera", "rapida", "precisa", "duradera", "comoda",
        "tecnica", "potente", "firme", "suave", "segura",
    ]
    text = " ".join(["snowboard"] * 9 + fillers * 5)
    word, ratio = keyword_stuffing_ratio(text)
    assert word == "snowboard"
    # ~8.3%: habria fallado con el umbral viejo de 8%, pasa con el de 15%.
    assert 0.08 < ratio < MAX_STUFFING_RATIO


# ---------------------------------------------------------------------
# Ancho en pixeles (informativo)
# ---------------------------------------------------------------------

def test_pixel_width_titulo_largo_mayor_que_corto():
    corto = estimate_title_pixel_width("Tabla Pro")
    largo = estimate_title_pixel_width(
        "Tabla de Snowboard Pro Series para Montaña Todo Terreno Extremo"
    )
    assert largo > corto


# ---------------------------------------------------------------------
# Evaluacion completa: los casos reales que motivaron este modulo
# ---------------------------------------------------------------------

def test_evaluate_rechaza_titulo_colapsado_tipo_tabla_pro():
    """Caso real: el producto 1009 degradado a 'Tabla Pro Extreme'."""
    result = evaluate_deterministic(
        title="Tabla Pro Extreme",
        body_html="<p>Tabla de alta calidad para todos los niveles.</p>",
    )
    assert result.passes_gate is False
    assert any("corto" in f.lower() for f in result.failures)


def test_evaluate_aprueba_titulo_con_hechos_y_estructura():
    """El texto bueno del 1009 restaurado (All-Mountain)."""
    result = evaluate_deterministic(
        title="Tabla All-Mountain de Alto Rendimiento | Control y Velocidad Total",
        body_html=(
            "<ul>"
            "<li><strong>Dominio en laderas:</strong> soporta pendientes de hasta 45 grados "
            "y se adapta a terrenos irregulares con facilidad.</li>"
            "<li><strong>Resistencia extrema:</strong> construida para climas de hasta -20 grados "
            "sin perder flexibilidad.</li>"
            "<li><strong>Medidas:</strong> 158 cm de largo, ideal para riders de 1.70 a 1.85 m.</li>"
            "</ul>"
        ),
    )
    assert result.title_length_ok is True
    assert result.has_structural_elements is True
    assert result.concrete_facts_count >= 2
    assert "corto" not in " ".join(result.failures).lower()


def test_evaluate_rechaza_palabra_prohibida():
    result = evaluate_deterministic(
        title="Tabla All-Mountain de Alto Rendimiento y Calidad Garantizada Total",
        body_html="<ul><li>Mide 158 cm y pesa 3 kg de material premium.</li></ul>",
        forbidden_words=["garantizada"],
    )
    assert result.passes_gate is False
    assert "garantizada" in result.forbidden_words_found


def test_evaluate_detecta_titulo_demasiado_largo():
    title_largo = "T" * (TITLE_MAX_LENGTH + 5)
    result = evaluate_deterministic(title=title_largo, body_html="<p>Texto de relleno.</p>")
    assert result.title_length_ok is False


def test_evaluate_titulo_en_banda_valida_no_falla_por_longitud():
    title_valido = "T" * TITLE_MIN_LENGTH
    result = evaluate_deterministic(title=title_valido, body_html="<p>x</p>")
    assert result.title_length_ok is True


def test_inflesz_es_informativo_no_eliminatorio():
    """
    Regresion 2026-08-15.

    El texto All-Mountain del 1009 es el mejor del catalogo real y saca
    INFLESZ 31.4 ("muy_dificil") por vocabulario tecnico polisilabico.
    El umbral de 55 del paper aplica explicitamente a "textos sobre
    salud"; en ese mismo estudio las revistas cientificas promediaron
    37.9. La legibilidad se mide y se reporta, pero NUNCA reprueba sola.
    """
    result = evaluate_deterministic(
        title="Tabla All-Mountain de Alto Rendimiento | Control y Velocidad Total",
        body_html=(
            "<ul>"
            "<li><strong>Dominio en laderas:</strong> se adapta al instante a "
            "cualquier terreno elevado y relieves tecnicos para una experiencia "
            "fluida y segura.</li>"
            "<li><strong>Resistencia extrema:</strong> construccion reforzada que "
            "soporta impactos en climas de heladas, garantizando una vida util "
            "prolongada.</li>"
            "<li><strong>Flexibilidad dinamica:</strong> el equilibrio perfecto "
            "entre rigidez para el descenso y maniobrabilidad tecnica.</li>"
            "</ul>"
        ),
    )
    assert result.inflesz_score is not None
    assert result.inflesz_band in {"muy_dificil", "algo_dificil"}
    assert not any("INFLESZ" in f for f in result.failures)
    assert result.passes_gate is True