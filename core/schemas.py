from pydantic import BaseModel, Field
from typing import List

# ==========================================
# 1. Lo que recibe el Agente (La realidad de Shopify)
# ==========================================
class ProductContext(BaseModel):
    shopify_id: str
    current_title: str
    current_body_html: str
    inventory_quantity: int
    sales_last_7_days: int
    
# ==========================================
# 2. Las reglas del Cliente (El Cerebro de Marca)
# ==========================================
class BrandRules(BaseModel):
    tone_voice: str
    target_audience: str
    language: str
    forbidden_words: List[str]
    brand_dna: str
    formatting_rules: str

# ==========================================
# 3. Lo que la IA está OBLIGADA a devolver (El Output del Escritor)
# ==========================================
class AIProposalOutput(BaseModel):
    new_title: str = Field(..., description="SEO optimized title. Max 70 characters.")
    new_body_html: str = Field(..., description="High conversion HTML description using <ul> and <strong> tags. No markdown code blocks.")
    seo_tags: str = Field(..., description="Comma separated SEO keywords.")
    audit_score: int = Field(..., ge=0, le=100, description="Health score from 0 to 100.")
    audit_log: List[str] = Field(..., description="List of 3 to 5 specific reasons why the original text was bad and how it was fixed.")

# ==========================================
# 4. Lo que el Juez está OBLIGADO a devolver (El Bucle de Crítica)
# ==========================================
class CriticFeedback(BaseModel):
    is_perfect: bool = Field(..., description="True ONLY if the copy meets absolutely ALL SEO and Brand rules. False if there is even one minor error.")
    issues_found: List[str] = Field(..., description="List of specific violations found (e.g. 'Used forbidden word: plastic', 'Title is 85 chars, needs to be under 70').")
    suggestions: str = Field(..., description="Surgical instructions for the writer agent to fix the copy in the next iteration.")