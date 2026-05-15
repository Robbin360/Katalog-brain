import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

# Initialize Google GenAI client
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("Listing available models...")
models = genai_client.models.list()

for model in models:
    print(f"Model: {model.name}, Supported Actions: {model.supported_actions}")
