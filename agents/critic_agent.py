from typing import Any

from pydantic_ai import Agent
from core.schemas import CriticFeedback
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


primary_critic = Agent(
    model='google-gla:gemini-3-flash-preview', 
    output_type=CriticFeedback,
    system_prompt=(
        "You are a $100B SaaS Quality Auditor. "
        "Your logic is binary: PERFECT or FAILED. "
        "You have zero tolerance for forbidden words or poor SEO. "
        "Your instructions to the writer must be surgical and direct."
    )
)

fallback_critic = Agent(
    model='google-gla:gemini-1.5-flash',
    output_type=CriticFeedback,
    system_prompt=(
        "You are a strict QA checker for Shopify product copy. "
        "Return FAILED unless all brand rules, forbidden-word rules, and SEO constraints pass. "
        "List concrete issues and short fix instructions. "
        "Return only valid structured data matching the schema."
    )
)

# Backward-compatible alias for older imports.
critic_agent = primary_critic


async def run_critic(prompt: str) -> Any:
    try:
        return await primary_critic.run(prompt)
    except Exception as error:
        if not _should_use_fallback(error):
            raise

        print(
            "⚠️ [Fallback] El Primary Critic falló por Cuota. "
            "Despertando al Fallback Agent (Flash)..."
        )
        return await fallback_critic.run(prompt)
