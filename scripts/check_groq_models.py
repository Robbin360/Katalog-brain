import os, httpx
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GROQ_API_KEY")

r = httpx.get(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
models = r.json().get("data", [])

print(f"{'ID':55s} | {'OWNED_BY'}")
print("=" * 80)
for m in sorted(models, key=lambda x: x["id"]):
    print(f"{m['id']:55s} | {m.get('owned_by', '?')}")

print("\n--- CANDIDATOS PESADOS (70B) ---")
for m in models:
    mid = m["id"]
    if "70b" in mid.lower() or "llama3" in mid.lower() or "llama-3" in mid.lower():
        print(f"  {m['id']:55s} | {m.get('owned_by', '?')}")
