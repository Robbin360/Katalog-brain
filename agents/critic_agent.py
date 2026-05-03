from pydantic_ai import Agent
from core.schemas import CriticFeedback
from dotenv import load_dotenv

load_dotenv()

critic_agent = Agent(
    # Usamos Flash para el Juez (es ultrarrápido y barato para auditar)
    model='google-gla:gemini-3-flash-preview', 
    output_type=CriticFeedback,
    system_prompt=(
        "You are a $100B SaaS Quality Auditor. "
        "Your logic is binary: PERFECT or FAILED. "
        "You have zero tolerance for forbidden words or poor SEO. "
        "Your instructions to the writer must be surgical and direct."
    )
)