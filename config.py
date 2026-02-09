import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Laravel
    LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api/v1/ia")

    # Configuración del sistema
    MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS", 6))
    PRODUCTS_TO_RETURN = int(os.getenv("PRODUCTS_TO_RETURN", 16))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.4))

    # Modelo de embeddings (local, gratis)
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # LLM
    LLM_MODEL = "llama-3.3-70b-versatile"  # Groq (gratis y rápido)

    # ChromaDB
    CHROMA_PATH = "./data/chroma_db"

config = Config()