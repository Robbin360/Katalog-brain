import os
import sys
from dotenv import load_dotenv
from google import genai

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 70)
print("GOOGLE EMBEDDING MODELS — DISCOVERY REPORT")
print("=" * 70)

for m in client.models.list():
    actions = m.supported_actions if hasattr(m, 'supported_actions') else []
    if 'embedContent' not in actions:
        continue

    print(f"\n[Model] {m.name}")
    print(f"   Description: {getattr(m, 'description', 'N/A')[:120]}")

    dims = "N/A"
    try:
        result = client.models.embed_content(
            model=m.name, contents="test"
        )
        vec = result.embeddings[0].values
        dims = len(vec)
        print(f"   [OK] Dimensions: {dims}")
    except Exception as e:
        print(f"   [FAIL] Embed test error: {str(e)[:80]}")

print("\n" + "=" * 70)
print("RECOMMENDATION FOR SUPABASE:")
print("-" * 70)
print("Use the model with the highest dimensions >= 768.")
print("Set VECTOR(n) in Supabase to match the exact dimension.")
print("=" * 70)
