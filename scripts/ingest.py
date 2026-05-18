import os
import sys
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ingestion_agent import ingestion_agent
from core.schemas import KnowledgeChunk

load_dotenv()

# Initialize Google GenAI client
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Supabase client with service role key
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

def generate_embedding(text: str) -> list[float]:
    """Usa models/gemini-embedding-2 con truncamiento Matryoshka a 1536."""
    result = genai_client.models.embed_content(
        model='models/gemini-embedding-2',
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=1536)
    )
    return result.embeddings[0].values

def save_to_knowledge_base(chunk: KnowledgeChunk, embedding: List[float]) -> bool:
    """Save a knowledge chunk with its embedding to Supabase."""
    try:
        response = supabase.table('knowledge_base').insert({
            "content": chunk.content,
            "embedding": embedding,
            "metadata": {
                "category": chunk.category,
                "tags": chunk.tags,
                "source": chunk.source,
                "ecommerce_applicability": chunk.ecommerce_applicability,
                "engine": "gemini-embedding-2"
            }
        }).execute()
        
        print("   [DB] Saved chunk: " + chunk.content[:50] + "...")
        return True
    except Exception as e:
        print("   [ERROR] Error saving to knowledge_base: " + str(e))
        return False

def process_text(text: str):
    """Process raw text through the ingestion pipeline."""
    print("[START] Starting ingestion pipeline...")
    print("[INFO] Processing text (" + str(len(text)) + " characters)...")
    
    # Step 1: Extract knowledge chunks using the ingestion agent
    try:
        print("[AI] Extracting knowledge chunks with AI agent...")
        result = ingestion_agent.run_sync(text)
        # Handle different result structures from PydanticAI
        ingestion_result = getattr(result, 'data', getattr(result, 'output', result))
        chunks: List[KnowledgeChunk] = ingestion_result.chunks
        print("[SUCCESS] Extracted " + str(len(chunks)) + " knowledge chunks")
    except Exception as e:
        print("[ERROR] Error extracting chunks: " + str(e))
        return
    
    # Step 2: Generate embeddings and save to database
    print("[EMBED] Generating embeddings and saving to database...")
    success_count = 0
    
    for i, chunk in enumerate(chunks, 1):
        print("\n[" + str(i) + "/" + str(len(chunks)) + "] Processing chunk...")
        print("   Category: " + repr(chunk.category))
        print("   Tags: " + repr(', '.join(chunk.tags)))
        print("   Source: " + repr(chunk.source))
        print("   Applicability: " + repr(chunk.ecommerce_applicability))
        
        try:
            # Generate embedding
            print("   [STEP] Generating embedding...")
            embedding = generate_embedding(chunk.content)
            if embedding:
                print("   [SUCCESS] Embedding generated (dimension: " + str(len(embedding)) + ")")
            else:
                print("   [ERROR] Embedding is None")
                continue
            
            # Save to database
            print("   [STEP] Saving to database...")
            if save_to_knowledge_base(chunk, embedding):
                success_count += 1
        except Exception as e:
            print("   [ERROR] Failed to process chunk: " + str(e))
            import traceback
            traceback.print_exc()
            continue
    
    print("\n[DONE] Ingestion complete! Successfully saved " + str(success_count) + "/" + str(len(chunks)) + " chunks to knowledge_base.")

def main():
    """Main execution function with test data."""
    # Test text about sales psychology
    test_text = """
    Contexto 2026: Títulos para la Era de IA
Este bloque es crítico para Katalog AI dado el panorama actual:
El problema:
Los AI Overviews reducen clics un 58% (Ahrefs, febrero 2026). El 60% de las búsquedas termina sin clic. Solo el 1% de las búsquedas llevan a que el usuario haga clic en un enlace dentro del AI Overview. Position Digital
La oportunidad:
Cuando tu marca es citada dentro del AI Overview, el CTR orgánico es 35% más alto. Los AI Overviews aparecen en el 99.9% de keywords informacionales, pero solo en el 3.2% de búsquedas de Shopping — el canal de e-commerce está relativamente protegido. Position Digital
Implicación para títulos en 2026:

Shopping está protegido. Los títulos de producto en Google Shopping tienen menor interferencia de IA que el contenido informacional.
Títulos para ser citados. Un título debe ser lo suficientemente claro y específico para que un LLM lo cite como fuente autoritativa.
Intención transaccional primero. Los sectores con menor presencia de AI Overviews son Shopping (3.2%) y Real Estate (5.8%) — el e-commerce mantiene CTR orgánico alto. 
    """
    
    process_text(test_text)

if __name__ == "__main__":
    main()
