from pydantic import BaseModel, Field
from typing import List, Optional, Any, Literal

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
    new_title: str = Field(..., description="SEO optimized title. Max 70 characters.")
    new_body_html: str = Field(..., description="High conversion HTML description using <ul> and <strong> tags.")
    seo_tags: str = Field(..., description="Comma separated SEO keywords.")
    audit_score: int = Field(..., ge=0, le=100)
    audit_log: List[str] = Field(..., description="Specific reasons for changes.")

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
    platform_compatibility: List[Literal['shopify', 'amazon', 'all']]
    target_niche: List[str]
    price_bracket: Literal['budget', 'mid-range', 'luxury', 'all']
    business_model: Literal['d2c', 'dropshipping', 'brand-luxury', 'all']
    locale: List[str]
    market_maturity: Literal['educated', 'uneducated']
    buyer_archetype: Literal['impulse', 'researcher', 'deal_hunter', 'skeptic']
    funnel_stage: Literal['awareness', 'consideration', 'conversion', 'retention', 'upsell']
    primary_trigger: str
    content_placement: Literal['title', 'description_body', 'bullet_points', 'meta_tags', 'all']
    content_length_target: Literal['micro', 'short', 'medium', 'long', 'all']
    evidence_basis: Literal['ab_test_large', 'expert_opinion', 'case_study', 'academic_study']
