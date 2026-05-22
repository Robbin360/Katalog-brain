from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()

# 1. EL MOLDE ESTRICTO (Lo que la IA está obligada a responder)
class InspectorResult(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Puntuación de salud SEO y Ventas (0-100).")
    reason: str = Field(..., description="Máximo 15 palabras. Razón brutalmente honesta de por qué el texto vende o fracasa.")

# 2. EL AGENTE (Rápido, barato y letal)
inspector_agent = Agent(
    model='google-gla:gemini-3.5-flash', # <-- El modelo ultrarrápido
    output_type=InspectorResult,
    system_prompt=(
        "You are a ruthless E-commerce Conversion Rate (CRO) and SEO Auditor. "
        "I will give you a Shopify product title and its HTML description. "
        "Evaluate it strictly on: "
        "1. Emotional hook (Does it make me want to buy?) "
        "2. SEO clarity (Are there clear keywords?) "
        "3. Readability (Is it a wall of text or well-formatted?). "
        "Return a score from 0 to 100 and a very short, punchy reason for your score."
    )
)