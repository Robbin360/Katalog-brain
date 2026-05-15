from pydantic_ai import Agent
from core.schemas import IngestionResult
from dotenv import load_dotenv

load_dotenv()

ingestion_agent = Agent(
    model='google-gla:gemini-3-flash-preview',
    output_type=IngestionResult,
    system_prompt=(
        "Eres un analista de marketing B2B y experto en CRO. "
        "Extrae reglas de oro accionables del texto proporcionado. "
        "Ignora introducciones o paja. "
        "Devuelve conceptos densos, independientes y listos para ser usados "
        "por una IA que optimiza productos de e-commerce."
    )
)
