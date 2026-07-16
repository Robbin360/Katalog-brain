import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

INSPECTOR_MODEL = os.getenv("INSPECTOR_MODEL", "google-gla:gemini-3.1-flash-lite")
OPTIMIZER_PRIMARY_MODEL = os.getenv("OPTIMIZER_PRIMARY_MODEL", "google-gla:gemini-2.5-pro")
OPTIMIZER_FALLBACK_MODEL = os.getenv("OPTIMIZER_FALLBACK_MODEL", "groq:llama-3.3-70b-versatile")
CRITIC_PRIMARY_MODEL = os.getenv("CRITIC_PRIMARY_MODEL", "google-gla:gemini-2.5-flash")
CRITIC_FALLBACK_MODEL = os.getenv("CRITIC_FALLBACK_MODEL", "groq:openai/gpt-oss-120b")

RECLASSIFIER_PRIMARY_MODEL = os.getenv("RECLASSIFIER_PRIMARY_MODEL", "deepseek/deepseek-v4-flash:free")
RECLASSIFIER_FALLBACK_MODEL = os.getenv("RECLASSIFIER_FALLBACK_MODEL", "groq:qwen/qwen3-32b")

# Motor ultrarrápido para el Agente Investigador (Tavily/Firecrawl)
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "google:gemini-3.5-flash")