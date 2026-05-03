import os
from pydantic_ai import Agent
from core.schemas import AIProposalOutput
from dotenv import load_dotenv

load_dotenv()

# Instanciamos el Agente con Gemini 3.1 Pro
# PydanticAI se encarga automáticamente de forzar la salida para que encaje en AIProposalOutput
optimizer_agent = Agent(
    model='google-gla:gemini-3-flash-preview', # Conectando al modelo de frontera
    output_type=AIProposalOutput,
    system_prompt=(
        "You are a $100M/year E-commerce Conversion Rate Optimization (CRO) expert. "
        "Your mission is to rewrite Shopify product listings to maximize sales revenue. "
        "You must STRICTLY adhere to the brand's DNA, formatting rules, and NEVER use forbidden words. "
        "You do not write fluff. You write high-converting, technical, and benefit-driven copy. "
        "Output ONLY valid structured data matching the requested schema."
    )
)