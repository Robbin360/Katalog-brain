import os
import sys
from typing import List
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure stdout to use UTF-8 on Windows to avoid UnicodeEncodeError with emojis
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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
        chunk_metadata = chunk.model_dump(exclude={'content'})
        chunk_metadata['engine'] = 'gemini-embedding-2-1536'
        
        data = {
            "content": chunk.content,
            "embedding": embedding,
            "metadata": chunk_metadata
        }
        response = supabase.table('knowledge_base').insert(data).execute()
        
        print("   [DB] Saved chunk: " + chunk.content[:50] + "...")
        return True
    except Exception as e:
        print("   [ERROR] Error saving to knowledge_base: " + str(e))
        return False

def process_text(text: str) -> bool:
    """Process raw text through the ingestion pipeline. Returns True if completely successful with no errors."""
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
        return False
    
    if not chunks:
        print("[WARNING] No chunks were extracted from the text.")
        return False
    
    # Step 2: Generate embeddings and save to database
    print("[EMBED] Generating embeddings and saving to database...")
    success_count = 0
    has_errors = False
    
    for i, chunk in enumerate(chunks, 1):
        print("\n[" + str(i) + "/" + str(len(chunks)) + "] Processing chunk...")
        print(f"   🎯 Trigger: {chunk.primary_trigger} | Funnel: {chunk.funnel_stage}")
        print("   Tags: " + repr(', '.join(chunk.tags)))
        print("   Source: " + repr(chunk.source))
        
        try:
            # Generate embedding
            print("   [STEP] Generating embedding...")
            embedding = generate_embedding(chunk.content)
            if embedding:
                print("   [SUCCESS] Embedding generated (dimension: " + str(len(embedding)) + ")")
            else:
                print("   [ERROR] Embedding is None")
                has_errors = True
                continue
            
            # Save to database
            print("   [STEP] Saving to database...")
            if save_to_knowledge_base(chunk, embedding):
                success_count += 1
            else:
                has_errors = True
        except Exception as e:
            print("   [ERROR] Failed to process chunk: " + str(e))
            import traceback
            traceback.print_exc()
            has_errors = True
            continue
    
    print("\n[DONE] Ingestion complete! Successfully saved " + str(success_count) + "/" + str(len(chunks)) + " chunks to knowledge_base.")
    return not has_errors and success_count == len(chunks)

def main():
    """Main execution function reading from knowledge.txt."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    knowledge_path = os.path.join(root_dir, "knowledge.txt")
    
    if not os.path.exists(knowledge_path):
        print("💤 El buzón knowledge.txt está vacío. Nada que procesar.")
        return
        
    try:
        with open(knowledge_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        print("[ERROR] Error al leer el archivo knowledge.txt: " + str(e))
        return
        
    if len(content.strip()) < 20:
        print("💤 El buzón knowledge.txt está vacío. Nada que procesar.")
        return
        
    # Process text
    success = process_text(content)
    
    if success:
        try:
            with open(knowledge_path, "w", encoding="utf-8") as f:
                f.write("")
            print("🏁 [DONE] Ingesta completada. El archivo knowledge.txt ha sido vaciado.")
        except Exception as e:
            print("[ERROR] No se pudo vaciar el archivo knowledge.txt: " + str(e))

if __name__ == "__main__":
    main()
