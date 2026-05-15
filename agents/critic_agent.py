from typing import Any

from pydantic_ai import Agent
from core.schemas import CriticFeedback
from dotenv import load_dotenv

load_dotenv()

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
    model='groq:llama-3.1-8b-instant',
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


async def run_critic_with_fallback(prompt: str) -> Any:
    try:
        return await primary_critic.run(prompt)
    except Exception as e:
        print(
            "⚠️ [Fallback] Error en Google API (Quota/Timeout). "
            "Activando motor Groq Llama 3..."
        )
        return await fallback_critic.run(prompt)
