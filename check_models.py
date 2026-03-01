import os
from dotenv import load_dotenv
load_dotenv()
from google import genai

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

print("=" * 60)
print("ALL AVAILABLE MODELS")
print("=" * 60)

embedding_models = []
generation_models = []

for model in client.models.list():
    name = model.name
    supported = getattr(model, "supported_actions", []) or getattr(model, "supported_generation_methods", [])
    supported_str = str(supported)
    
    if "embed" in name.lower() or "embedding" in supported_str.lower():
        embedding_models.append(name)
        print(f"[EMBED]  {name}")
    elif "gemini" in name.lower() or "flash" in name.lower() or "pro" in name.lower():
        generation_models.append(name)
        print(f"[GEN]    {name}")

print()
print("=" * 60)
print("EMBEDDING MODELS ONLY")
print("=" * 60)
for m in embedding_models:
    print(f"  -> {m}")

print()
print("=" * 60)
print("GENERATION MODELS ONLY")
print("=" * 60)
for m in generation_models:
    print(f"  -> {m}")
