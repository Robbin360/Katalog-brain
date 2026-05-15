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
                "tags": chunk.tags
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
        # print("   Content: " + chunk.content[:100] + "...")  # Temporarily disabled
        
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
    La urgencia y la escasez son principios fundamentales de la psicología de ventas. 
    Cuando un cliente percibe que una oferta está limitada en tiempo o cantidad, 
    su cerebro activa mecanismos de pérdida aversión, aumentando significativamente 
    la probabilidad de compra. Los contadores regresivos, las etiquetas de 
    "últimas unidades" y las ofertas por tiempo limitado son tácticas probadas 
    que capitalizan este sesgo cognitivo. Sin embargo, es crucial usar estas 
    técnicas éticamente y con genuina limitación para mantener la confianza del cliente.
    
    El principio de reciprocidad dicta que cuando recibimos algo de valor, 
    sentimos una obligación psicológica de corresponder. En e-commerce, esto 
    se traduce en ofrecer contenido gratuito, muestras, o consultas iniciales 
    antes de pedir una venta. Los clientes que reciben valor primero son 
    significativamente más propensos a comprar.
    
    La prueba social es otro poderoso factor de influencia. Mostrar testimonios, 
    reseñas de clientes, números de ventas y casos de éxito activa el deseo 
    de conformidad social. Los compradores miran el comportamiento de otros 
    para validar sus decisiones, especialmente en productos de mayor valor.
    """
    
    process_text(test_text)

if __name__ == "__main__":
    main()
