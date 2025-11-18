import os
from dotenv import load_dotenv

# Check if running on Hugging Face Spaces
is_huggingface_space = os.getenv("SPACE_ID") is not None

# Only load .env file when running locally, not on HF Spaces
if not is_huggingface_space:
    load_dotenv(override=True)

# Get configuration values (works for both local .env and HF Spaces secrets)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
DB_PATH = "research_agent.db"
MAX_CHUNK_SIZE = 12000  # Increased for better full-document processing
OVERLAP_SIZE = 200

# Available LLM models (OpenAI only)
LLM_MODELS = {
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"}
}
