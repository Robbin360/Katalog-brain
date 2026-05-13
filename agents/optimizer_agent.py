from typing import Any

from pydantic_ai import Agent
from core.schemas import AIProposalOutput
from dotenv import load_dotenv

load_dotenv()

FALLBACK_ERROR_HINTS = (
    "429",
    "quota",
    "rate limit",
    "resource exhausted",
    "modelhttperror",
    "status_code",
    "google",
    "genai",
    "overloaded",
    "temporarily unavailable",
    "service unavailable",
)


def _should_use_fallback(error: Exception) -> bool:
    error_text = f"{type(error).__name__}: {error}".lower()
    return any(hint in error_text for hint in FALLBACK_ERROR_HINTS)


primary_optimizer = Agent(
    model='google-gla:gemini-3.1-pro-preview',
    output_type=AIProposalOutput,
    system_prompt=(
        "You are a $100M/year E-commerce Conversion Rate Optimization (CRO) expert. "
        "Your mission is to rewrite Shopify product listings to maximize sales revenue. "
        "You must STRICTLY adhere to the brand's DNA, formatting rules, and NEVER use forbidden words. "
        "You do not write fluff. You write high-converting, technical, and benefit-driven copy. "
        "Output ONLY valid structured data matching the requested schema."
    )
)

fallback_optimizer = Agent(
    model='google-gla:gemini-3-flash-preview',
    output_type=AIProposalOutput,
    system_prompt=(
        "You are a strict Shopify product copy optimizer. "
        "Rewrite the listing with clear benefits, compliant SEO, and concise conversion copy. "
        "Follow every brand rule exactly. Never use forbidden words. "
        "Keep the title under 70 characters. "
        "Return only valid structured data matching the schema."
    )
)

# Backward-compatible alias for older imports.
optimizer_agent = primary_optimizer


async def run_optimizer(prompt: str) -> Any:
    try:
        return await primary_optimizer.run(prompt)
    except Exception as error:
        if not _should_use_fallback(error):
            raise

        print(
            "⚠️ [Fallback] El Primary Optimizer (Pro) falló por Cuota. "
            "Despertando al Fallback Agent (Flash)..."
        )
        return await fallback_optimizer.run(prompt)
