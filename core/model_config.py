import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

load_dotenv()

INSPECTOR_MODEL = os.getenv("INSPECTOR_MODEL", "google-gla:gemini-3.1-flash-lite")
OPTIMIZER_PRIMARY_MODEL = os.getenv("OPTIMIZER_PRIMARY_MODEL", "google-gla:gemini-3.5-flash")
# Groq retiró llama-3.3-70b-versatile el 2026-08-16 (404 model_not_found).
# Reemplazo oficial recomendado por Groq: openai/gpt-oss-120b, el mismo que
# ya usa CRITIC_FALLBACK_MODEL con éxito. Verificado tras un fallo real en
# el producto 1010 el 2026-08-17.
OPTIMIZER_FALLBACK_MODEL = os.getenv("OPTIMIZER_FALLBACK_MODEL", "groq:openai/gpt-oss-120b")
CRITIC_PRIMARY_MODEL = os.getenv("CRITIC_PRIMARY_MODEL", "google-gla:gemini-2.5-flash")
CRITIC_FALLBACK_MODEL = os.getenv("CRITIC_FALLBACK_MODEL", "groq:openai/gpt-oss-120b")

RECLASSIFIER_PRIMARY_MODEL = os.getenv("RECLASSIFIER_PRIMARY_MODEL", "deepseek/deepseek-v4-flash:free")
# Groq retiró qwen/qwen3-32b el 2026-07-17 (ver console.groq.com/docs/deprecations).
# Reemplazo oficial recomendado por Groq: openai/gpt-oss-120b, el mismo que
# ya usan los fallbacks del Optimizer y el Critic.
RECLASSIFIER_FALLBACK_MODEL = os.getenv("RECLASSIFIER_FALLBACK_MODEL", "groq:openai/gpt-oss-120b")

# Motor ultrarrápido para el Agente Investigador (Tavily/Firecrawl)
RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "google:gemini-3.5-flash")

# El orquestador tenía el modelo hardcodeado en agents/orchestrator_agent.py.
# Sin fallback, un 503 de Gemini lo degradaba en silencio al plan conservador.
ORCHESTRATOR_PRIMARY_MODEL = os.getenv("ORCHESTRATOR_PRIMARY_MODEL", "google:gemini-3.5-flash")
ORCHESTRATOR_FALLBACK_MODEL = os.getenv("ORCHESTRATOR_FALLBACK_MODEL", "groq:openai/gpt-oss-120b")