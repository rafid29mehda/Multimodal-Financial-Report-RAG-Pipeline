import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "SAMSUNG ELECTRONICS_4Q_financial_report.pdf"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")

# ---- API Keys ----
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ---- Model Config ----
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
CHROMA_COLLECTION = "samsung_financial_report"

# ---- Retrieval Config ----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
ID_KEY = "doc_id"
