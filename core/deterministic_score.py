"""
core/deterministic_score.py

Puntaje determinista de calidad de copy, SIN IA y SIN red.
Objetivo: medir lo medible con procesos oficiales o reglas lingüísticas
verificables, dejando a la IA (inspector_agent) solo lo irreducible:
persuasión y corrección semántica del lenguaje.

Diseño deliberado: esto es una PUERTA (pass/fail), no una función a
maximizar. Una métrica que se maximiza se explota (Ley de Goodhart):
si "más corto = más puntos", el writer converge a "Tabla Pro".
Por eso casi todo aquí son bandas con piso Y techo, no solo un mínimo.

CALIBRADO CONTRA EL CATÁLOGO REAL (18 productos, 2026-08-15):
la primera versión rechazaba 17 de 18. Ver notas en cada umbral.

Fuentes:
- Escala INFLESZ (adaptación española de Flesch-Szigriszt), validada
  sobre 210 textos aleatorios: Barrio-Cantalejo et al., "Validación de
  la Escala INFLESZ para evaluar la legibilidad de los textos dirigidos
  a pacientes", Anales Sis San Navarra, 2008;31(2):135-152.
  Fórmula: I = 206.835 - 62.3*(sílabas/palabras) - (palabras/frases)
  Bandas: <40 muy difícil | 40-55 algo difícil | 55-65 normal |
          65-80 bastante fácil | >80 muy fácil
- Longitud de título: Google NO publica un límite de caracteres
  (Google Search Central, "Influencing your title links"). Trunca por
  ANCHO EN PÍXELES con fuente proporcional, no por conteo de letras
  (confirmado por mediciones independientes: Zyppy ~580-600px /
  50-60 caracteres en desktop, hasta 70-80 en mobile). El techo de 70
  caracteres usado aquí coincide con el límite YA impuesto por
  core/schemas.py:AIProposalOutput.new_title (max_length=70), no es un
  número nuevo inventado. El piso de 40 SÍ es una regla de producto
  nuestra (no un estándar externo): es la línea que separa un título
  con categoría+beneficio real de un colapso tipo "Tabla Pro" (9 chars).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────
# 1. UTILIDADES DE TEXTO (sin dependencias externas)
# ─────────────────────────────────────────────────────────────────────

_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_WHITESPACE_RE = re.compile(r"\s+")

# Insertamos un punto al cerrar bloques estructurales ANTES de quitar
# las etiquetas, para que una lista de bullets sin puntuación no se
# colapse en una sola "frase" gigante (eso rompería el denominador F
# de la fórmula INFLESZ y daría una legibilidad falsamente baja).
_BLOCK_CLOSE_RE = re.compile(r"</(li|p|h1|h2|h3|h4)>", re.IGNORECASE)


def strip_html_to_text(html: str | None) -> str:
    """HTML -> texto plano, con puntos insertados en cierres de bloque."""
    if not html:
        return ""
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _BLOCK_CLOSE_RE.sub(". ", text)
    text = _TAG_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


_WORD_RE = re.compile(r"[a-zA-ZáéíóúÁÉÍÓÚñÑüÜ]+")


def count_words_es(text: str) -> int:
    """Cuenta tokens con al menos una letra. No cuenta números sueltos."""
    return len(_WORD_RE.findall(text))


_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def count_sentences_es(text: str) -> int:
    """Cuenta frases por puntuación terminal. Nunca devuelve 0."""
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return max(len(parts), 1)


# ─────────────────────────────────────────────────────────────────────
# 2. CONTEO DE SÍLABAS EN ESPAÑOL (reglas de diptongo/hiato)
# ─────────────────────────────────────────────────────────────────────
#
# Vocales fuertes (abiertas):  a e o (y sus tildes á é ó)
# Vocales débiles (cerradas):  i u ü (y sus tildes í ú)
#
# Regla de la Real Academia Española, aplicada aquí de forma explícita:
#   - fuerte + débil (sin tilde en la débil) o viceversa  -> DIPTONGO (1 sílaba)
#   - débil + débil (sin tilde)                            -> DIPTONGO (1 sílaba)
#   - fuerte + fuerte                                      -> HIATO (2 sílabas)
#   - cualquier combinación donde la débil lleva tilde (í/ú) -> HIATO (2 sílabas)
#     (la tilde "rompe" el diptongo: día, país, María)
#
# Verificado manualmente contra: día=2, aire=2, cae=2, cielo=2, país=2,
# bueno=2, María=3. Ver tests para los casos exactos.

_STRONG_VOWELS = set("aeoáéó")
_WEAK_VOWELS = set("iuü")
_WEAK_ACCENTED = set("íú")
_ALL_VOWELS = _STRONG_VOWELS | _WEAK_VOWELS | _WEAK_ACCENTED

_VOWEL_CLUSTER_RE = re.compile(f"[{''.join(_ALL_VOWELS)}]+", re.IGNORECASE)


def _syllables_in_cluster(cluster: str) -> int:
    """Cuenta cuántos núcleos silábicos hay en un grupo vocálico contiguo."""
    if len(cluster) <= 1:
        return 1
    count = 1
    for i in range(1, len(cluster)):
        prev_c, cur_c = cluster[i - 1], cluster[i]
        prev_weak_accented = prev_c in _WEAK_ACCENTED
        cur_weak_accented = cur_c in _WEAK_ACCENTED
        prev_strong = prev_c in _STRONG_VOWELS
        cur_strong = cur_c in _STRONG_VOWELS

        if prev_weak_accented or cur_weak_accented:
            is_hiatus = True
        elif prev_strong and cur_strong:
            is_hiatus = True
        else:
            is_hiatus = False  # diptongo: fuerte+débil, débil+fuerte, débil+débil

        if is_hiatus:
            count += 1
        # si es diptongo, no sumamos: se fusiona con el núcleo anterior
    return count


def count_syllables_word_es(word: str) -> int:
    """Cuenta sílabas de una sola palabra española."""
    word = word.lower()
    clusters = _VOWEL_CLUSTER_RE.findall(word)
    if not clusters:
        return 0
    return sum(_syllables_in_cluster(c) for c in clusters)


def count_syllables_es(text: str) -> int:
    """Cuenta sílabas totales de un texto (suma por palabra)."""
    words = _WORD_RE.findall(text)
    return sum(count_syllables_word_es(w) for w in words)


# ─────────────────────────────────────────────────────────────────────
# 3. LEGIBILIDAD: ÍNDICE DE FLESCH-SZIGRISZT / ESCALA INFLESZ
# ─────────────────────────────────────────────────────────────────────

INFLESZ_BANDS = (
    (40, "muy_dificil"),
    (55, "algo_dificil"),
    (65, "normal"),
    (80, "bastante_facil"),
    (float("inf"), "muy_facil"),
)


def flesch_szigriszt_inflesz(text: str) -> float | None:
    """
    I = 206.835 - 62.3*(S/P) - (P/F)
    S=sílabas, P=palabras, F=frases.
    Devuelve None si el texto es demasiado corto para medir con sentido
    (menos de 10 palabras): con tan poco texto la fórmula es ruido puro.
    """
    words = count_words_es(text)
    if words < 10:
        return None
    syllables = count_syllables_es(text)
    sentences = count_sentences_es(text)
    return 206.835 - 62.3 * (syllables / words) - (words / sentences)


def inflesz_band(score: float | None) -> str:
    if score is None:
        return "indeterminado"
    for threshold, name in INFLESZ_BANDS:
        if score < threshold:
            return name
    return "muy_facil"


# ─────────────────────────────────────────────────────────────────────
# 4. DENSIDAD DE HECHOS CONCRETOS
# ─────────────────────────────────────────────────────────────────────
# Distingue "Tabla All-Mountain de Alto Rendimiento" (con hechos) de
# "Tabla Pro" (genérico). Busca número + unidad reconocible.

_UNIT_TOKENS = (
    r"cm|mm|m|km|kg|g|mg|l|ml|w|kw|v|hz|db|%|"
    r"años?|meses?|d[ií]as?|grados?|°c|°f"
)
_CONCRETE_FACT_RE = re.compile(
    rf"\b\d+([.,]\d+)?\s*(?:{_UNIT_TOKENS})\b",
    re.IGNORECASE,
)


def count_concrete_facts(text: str) -> int:
    return len(_CONCRETE_FACT_RE.findall(text))


# ─────────────────────────────────────────────────────────────────────
# 5. ESTRUCTURA HTML (evitar "muro de texto")
# ─────────────────────────────────────────────────────────────────────

_STRUCTURAL_TAGS_RE = re.compile(
    r"<(ul|ol|li|h1|h2|h3|h4)\b", re.IGNORECASE
)
_PARAGRAPH_RE = re.compile(r"<p\b", re.IGNORECASE)


def has_structural_elements(html: str) -> bool:
    if _STRUCTURAL_TAGS_RE.search(html or ""):
        return True
    return len(_PARAGRAPH_RE.findall(html or "")) >= 2


# ─────────────────────────────────────────────────────────────────────
# 6. PALABRAS PROHIBIDAS Y STUFFING DE KEYWORDS
# ─────────────────────────────────────────────────────────────────────

# Lista curada corta, no exhaustiva. Documentado como heurística.
_SPANISH_STOPWORDS = frozenset(
    "de la el en y a que un una los las con para por es su sus del al "
    "se lo más o como este esta ese esa nuestro nuestra tu tus".split()
)


def find_forbidden_words(text: str, forbidden_words: list[str]) -> list[str]:
    """Mismo criterio que agents/critic_agent.py:pre_critic_checks:
    substring case-insensitive, sin límites de palabra (consistencia
    deliberada con la regla ya existente en el repo)."""
    haystack = text.lower()
    return [
        w for w in forbidden_words
        if w.strip() and w.strip().lower() in haystack
    ]


def keyword_stuffing_ratio(text: str) -> tuple[str | None, float]:
    """Devuelve (palabra_mas_repetida, ratio) ignorando stopwords cortas."""
    words = [w.lower() for w in _WORD_RE.findall(text) if len(w) > 3]
    words = [w for w in words if w not in _SPANISH_STOPWORDS]
    if not words:
        return None, 0.0
    counts: dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    top_word, top_count = max(counts.items(), key=lambda kv: kv[1])
    return top_word, top_count / len(words)


# ─────────────────────────────────────────────────────────────────────
# 7. ANCHO EN PÍXELES (ESTIMACIÓN INFORMATIVA, NO ES PUERTA DE FALLO)
# ─────────────────────────────────────────────────────────────────────
# Aproximación deliberadamente simple. NO reproduce el renderizado real
# de Google (fuente Arial variable). Solo advierte del riesgo de
# truncamiento; nunca reprueba el título por sí sola.

_NARROW_CHARS = set("iIjltf.,:;'! ")
_WIDE_CHARS = set("mMWw%")


def estimate_title_pixel_width(title: str) -> int:
    total = 0
    for ch in title:
        if ch in _NARROW_CHARS:
            total += 4
        elif ch in _WIDE_CHARS:
            total += 10
        else:
            total += 7
    return total


PIXEL_TRUNCATION_RISK_THRESHOLD = 600  # ver Zyppy: ~580-600px desktop


# ─────────────────────────────────────────────────────────────────────
# 8. RESULTADO Y EVALUACIÓN COMPLETA
# ─────────────────────────────────────────────────────────────────────

TITLE_MIN_LENGTH = 40  # regla de producto propuesta, NO estándar externo
TITLE_MAX_LENGTH = 70  # coincide con schemas.py:AIProposalOutput (existente)
BODY_MIN_WORDS = 25    # regla de producto propuesta, NO estándar externo

# Stuffing: 15% tras medir el catálogo real. Al 8% se marcaba 'snowboard'
# en productos de snowboard, es decir la palabra de la categoría. Falso
# positivo garantizado en catálogos de nicho.
MAX_STUFFING_RATIO = 0.15
MIN_STUFFING_COUNT = 5

# INFLESZ es INFORMATIVO, no eliminatorio.
#
# Por qué cambió: medido contra los 18 productos reales, el umbral de 55
# rechazaba 10 de 18, incluido el mejor texto del catálogo (id 1009, que
# saca 31.4). El paper de Barrio-Cantalejo fija ese mínimo explícitamente
# "en el caso de textos sobre salud"; en ese mismo estudio las revistas
# científicas promediaron 37.9. El español técnico de e-commerce es
# polisilábico ("resistencia" 4 sílabas, "maniobrabilidad" 6), así que
# cae estructuralmente en la banda 40-55. Usarlo como puerta castigaba
# precisión técnica, que es justo lo que queremos premiar.
#
# Se conserva medido y reportado para detectar tendencias, no para vetar.
ACCEPTABLE_INFLESZ_BANDS = {"normal", "bastante_facil", "muy_facil"}


@dataclass(frozen=True)
class DeterministicScoreResult:
    title_length: int
    title_length_ok: bool
    title_pixel_width_estimate: int
    title_truncation_risk: bool

    body_word_count: int
    body_sentence_count: int
    body_syllable_count: int
    body_word_count_ok: bool

    forbidden_words_found: list[str]
    concrete_facts_count: int
    has_structural_elements: bool

    inflesz_score: float | None
    inflesz_band: str

    top_repeated_word: str | None
    keyword_stuffing_ratio: float
    keyword_stuffing_flag: bool

    failures: list[str] = field(default_factory=list)

    @property
    def passes_gate(self) -> bool:
        return len(self.failures) == 0

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


def evaluate_deterministic(
    title: str | None,
    body_html: str | None,
    forbidden_words: list[str] | None = None,
) -> DeterministicScoreResult:
    title = title or ""
    body_html = body_html or ""
    forbidden_words = forbidden_words or []

    failures: list[str] = []

    # --- Título ---
    title_length = len(title)
    title_length_ok = TITLE_MIN_LENGTH <= title_length <= TITLE_MAX_LENGTH
    if title_length < TITLE_MIN_LENGTH:
        failures.append(
            f"Título demasiado corto ({title_length} chars, mínimo {TITLE_MIN_LENGTH})."
        )
    elif title_length > TITLE_MAX_LENGTH:
        failures.append(
            f"Título demasiado largo ({title_length} chars, máximo {TITLE_MAX_LENGTH})."
        )

    pixel_width = estimate_title_pixel_width(title)
    truncation_risk = pixel_width > PIXEL_TRUNCATION_RISK_THRESHOLD
    # Riesgo de truncamiento: informativo, no reprueba por sí solo.

    # --- Cuerpo ---
    body_text = strip_html_to_text(body_html)
    word_count = count_words_es(body_text)
    sentence_count = count_sentences_es(body_text)
    syllable_count = count_syllables_es(body_text)
    word_count_ok = word_count >= BODY_MIN_WORDS
    if not word_count_ok:
        failures.append(
            f"Descripción demasiado corta ({word_count} palabras, mínimo {BODY_MIN_WORDS})."
        )

    forbidden_found = find_forbidden_words(f"{title}\n{body_text}", forbidden_words)
    if forbidden_found:
        failures.append(f"Palabras prohibidas encontradas: {', '.join(forbidden_found)}.")

    facts_count = count_concrete_facts(body_text)
    structural = has_structural_elements(body_html)
    if facts_count == 0 and not structural:
        failures.append(
            "Sin hechos concretos (medidas/cifras) ni estructura (listas/encabezados): "
            "texto genérico tipo 'Tabla Pro'."
        )

    # Legibilidad: se mide y se reporta, pero NO reprueba. Ver comentario
    # en ACCEPTABLE_INFLESZ_BANDS.
    inflesz = flesch_szigriszt_inflesz(body_text)
    band = inflesz_band(inflesz)

    top_word, stuffing_ratio = keyword_stuffing_ratio(body_text)
    stuffing_count = int(stuffing_ratio * max(word_count, 1))
    stuffing_flag = stuffing_ratio > MAX_STUFFING_RATIO and stuffing_count >= MIN_STUFFING_COUNT
    if stuffing_flag:
        failures.append(
            f"Posible keyword stuffing: '{top_word}' se repite {stuffing_ratio:.0%} de las palabras."
        )

    return DeterministicScoreResult(
        title_length=title_length,
        title_length_ok=title_length_ok,
        title_pixel_width_estimate=pixel_width,
        title_truncation_risk=truncation_risk,
        body_word_count=word_count,
        body_sentence_count=sentence_count,
        body_syllable_count=syllable_count,
        body_word_count_ok=word_count_ok,
        forbidden_words_found=forbidden_found,
        concrete_facts_count=facts_count,
        has_structural_elements=structural,
        inflesz_score=inflesz,
        inflesz_band=band,
        top_repeated_word=top_word,
        keyword_stuffing_ratio=stuffing_ratio,
        keyword_stuffing_flag=stuffing_flag,
        failures=failures,
    )

    
    # ─────────────────────────────────────────────────────────────────────
# 9. PORCENTAJE PARA LA COLUMNA audit_score
# ─────────────────────────────────────────────────────────────────────
#
# La columna shopify_products.audit_score alimenta refresh_user_kpis
# (health_score_avg), get_priority_products (prioridad = stock*precio *
# (100-score)/100) y el dashboard. Antes la llenaba el inspector con una
# escala 0-100 sin calibrar; medido contra el catálogo real, esos valores
# estaban invertidos (el peor título del catálogo sacó 95).
#
# Este reemplazo es una SUMA DE CHECKS VERIFICABLES, no una opinión. Los
# pesos son una decisión de producto, no un estándar externo: lo
# defendible es que cada punto se rastrea a una comprobación concreta y
# que el resultado es idéntico en cada corrida.

QUALITY_WEIGHTS = {
    "title_length_ok": 25,
    "body_word_count_ok": 20,
    "has_structural_elements": 15,
    "no_forbidden_words": 10,
    "no_keyword_stuffing": 10,
    "concrete_facts_partial": 10,  # 1 o 2 datos concretos
    "concrete_facts_full": 20,     # 3 o más (reemplaza al parcial)
}


def quality_percent(result: DeterministicScoreResult) -> int:
    """
    Puntaje 0-100 derivado solo de checks deterministas.
    Un producto que cumple todo y trae 3+ datos concretos llega a 100.
    """
    score = 0

    if result.title_length_ok:
        score += QUALITY_WEIGHTS["title_length_ok"]
    if result.body_word_count_ok:
        score += QUALITY_WEIGHTS["body_word_count_ok"]
    if result.has_structural_elements:
        score += QUALITY_WEIGHTS["has_structural_elements"]
    if not result.forbidden_words_found:
        score += QUALITY_WEIGHTS["no_forbidden_words"]
    if not result.keyword_stuffing_flag:
        score += QUALITY_WEIGHTS["no_keyword_stuffing"]

    if result.concrete_facts_count >= 3:
        score += QUALITY_WEIGHTS["concrete_facts_full"]
    elif result.concrete_facts_count >= 1:
        score += QUALITY_WEIGHTS["concrete_facts_partial"]

    return min(score, 100)