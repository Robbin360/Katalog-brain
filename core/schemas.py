from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

# --- 1. CONTEXTO OPERATIVO (LangGraph Flow) ---
class ProductContext(BaseModel):
    shopify_id: str
    current_title: str
    current_body_html: str
    inventory_quantity: int
    sales_last_7_days: int

class BrandRules(BaseModel):
    tone_voice: str
    target_audience: str
    language: str
    forbidden_words: List[str]
    brand_dna: str
    formatting_rules: str

class AIProposalOutput(BaseModel):
    new_title: str = Field(..., max_length=70, description="SEO optimized title. Max 70 characters.")
    new_body_html: str = Field(..., min_length=80, description="High conversion HTML description using <ul> and <strong> tags.")
    seo_tags: str = Field(..., description="Comma separated SEO keywords.")
    audit_score: int = Field(..., ge=0, le=100)
    audit_log: List[str] = Field(..., min_length=1, description="Specific reasons for changes.")

class CriticFeedback(BaseModel):
    is_perfect: bool
    issues_found: List[str]
    suggestions: str

# --- 2. RAG KNOWLEDGE BASE (Evolutivo V4) ---
class KnowledgeChunk(BaseModel):
    content: str = Field(..., description="La regla de oro o concepto accionable.")
    source: str = Field(..., description="Origen: ej. 'Ogilvy 1972', 'Cialdini'.")
    
    # Dimensiones de Tienda y Producto
    platform_compatibility: List[str] = Field(..., description="['shopify', 'amazon', 'all']")
    target_niche: List[str] = Field(..., description="['fashion', 'electronics', 'home', 'all']")
    price_bracket: str = Field(..., description="['budget', 'mid-range', 'luxury', 'all']")
    business_model: str = Field(..., description="['d2c', 'dropshipping', 'brand-luxury']")
    
    # Psicología y Madurez de Mercado
    locale: List[str] = Field(..., description="['es_mx', 'es_es', 'es_latam', 'en_us', 'all']")
    market_maturity: str = Field(..., description="['educated', 'uneducated']")
    buyer_archetype: str = Field(..., description="['impulse', 'researcher', 'deal_hunter', 'skeptic']")
    
    # Estrategia de Conversión y Emoción
    funnel_stage: str = Field(..., description="['awareness', 'consideration', 'conversion', 'retention', 'upsell']")
    primary_trigger: str = Field(..., description="Ej: 'scarcity_ctr', 'authority_trust'.")
    
    # Formato y Calidad
    content_placement: str = Field(..., description="['title', 'description_body', 'bullet_points', 'all']")
    content_length_target: str = Field(..., description="['micro', 'short', 'medium', 'long']")
    evidence_basis: str = Field(..., description="['ab_test_large', 'expert_opinion', 'case_study', 'academic_study']")
    impact_weight: int = Field(..., ge=1, le=10)
    
    # Ciclo de Vida del Dato
    status: str = Field(default="active", description="['active', 'deprecated', 'under_review']")
    valid_until: Optional[str] = Field(None, description="Expiración (YYYY-MM)")
    supersedes: Optional[str] = Field(None, description="ID del chunk invalidado")
    
    # Métricas Evolutivas
    success_rate: float = 0.0
    usage_count: int = 0
    tags: List[str] = []

class IngestionResult(BaseModel):
    chunks: List[KnowledgeChunk]

class ReclassificationResult(BaseModel):
    """Campos a inferir por el Agente de Backfill. Independiente de KnowledgeChunk."""
    complexity_evaluation: str = Field(
        ...,
        exclude=True,
        description="Analisis inicial de la complejidad del fragmento antes de clasificar.",
    )
    complexity_level: Literal['LOW', 'MEDIUM', 'HIGH']
    deep_reasoning: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Razonamiento interno para casos MEDIUM/HIGH. No se persiste en Supabase.",
    )
    platform_compatibility: List[Literal['shopify', 'amazon', 'all']]
    target_niche: List[Literal['fashion', 'electronics', 'home', 'beauty', 'health', 'fitness', 'pets', 'food', 'all']]
    price_bracket: Literal['budget', 'mid-range', 'luxury', 'all']
    business_model: Literal['d2c', 'dropshipping', 'brand-luxury', 'all']
    locale: List[Literal['es_mx', 'es_es', 'es_latam', 'en_us', 'all']]
    market_maturity: Literal['educated', 'uneducated']
    buyer_archetype: Literal['impulse', 'researcher', 'deal_hunter', 'skeptic']
    funnel_stage: Literal['awareness', 'consideration', 'conversion', 'retention', 'upsell']
    primary_trigger: Literal['scarcity', 'social_proof', 'authority', 'reciprocity', 'logic', 'urgency']
    content_placement: Literal['title', 'description_body', 'bullet_points', 'meta_tags', 'all']
    content_length_target: Literal['micro', 'short', 'medium', 'long', 'all']
    evidence_basis: Literal['ab_test_large', 'expert_opinion', 'case_study', 'academic_study']
    impact_weight: int = Field(..., ge=1, le=10)

    @model_validator(mode='after')
    def validate_reasoning_depth(self) -> 'ReclassificationResult':
        reasoning = (self.deep_reasoning or '').strip()

        if self.complexity_level == 'HIGH' and len(reasoning) <= 150:
            raise ValueError('HIGH complexity requires deep_reasoning longer than 150 characters.')

        if self.complexity_level == 'MEDIUM' and len(reasoning) <= 50:
            raise ValueError('MEDIUM complexity requires deep_reasoning longer than 50 characters.')

        evaluation = self.complexity_evaluation.lower()
        complexity_terms = ('psicologia', 'psicología', 'arquetipo', 'sesgo')
        if self.complexity_level == 'LOW' and any(term in evaluation for term in complexity_terms):
            raise ValueError('Complexity cannot be LOW when evaluation mentions psychology, archetypes, or bias.')

        return self

    def to_supabase_dict(self) -> dict:
        """Return only distilled metadata; reasoning fields are excluded at field level."""
        return self.model_dump(exclude_none=True)
