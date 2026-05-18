from pydantic_ai import Agent
from core.schemas import IngestionResult
from dotenv import load_dotenv

load_dotenv()

ingestion_agent = Agent(
    model='google-gla:gemini-3-flash-preview',
    output_type=IngestionResult,
    system_prompt=(
        "Eres un Bibliotecario de E-commerce. "
        "Extrae reglas de oro accionables del texto proporcionado. "
        "Ignora introducciones o paja. Devuelve conceptos densos e independientes. "
        "Para cada regla, infiere el source (autor/libro si se menciona, sino 'General E-commerce'). "
        "Evalúa rigurosamente la ecommerce_applicability ('High', 'Medium', o 'Low') según el impacto directo en ventas. "
        "Clasifica cada regla en una category técnica exacta (ej. 'headline', 'trust', 'urgency', 'seo', "
        "'formatting', 'psychology', 'offer', 'pricing', 'call_to_action'). "
        "Devuelve conceptos listos para ser usados por una IA que optimiza productos de e-commerce."
    )
)
