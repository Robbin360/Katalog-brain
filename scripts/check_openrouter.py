import os, httpx, json
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENROUTER_API_KEY")

r = httpx.get(
    "https://openrouter.ai/api/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=30,
)
data = r.json()

print(f"{'ID':50s} | {'NAME'}")
print("=" * 100)
for m in data.get("data", []):
    mid = m["id"]
    if "free" in mid.lower():
        print(f"{mid:50s} | {m.get('name', '?')}")
