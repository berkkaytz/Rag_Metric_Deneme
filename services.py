# NOTE: API KEYS
# - Together API key is read from your environment via .env (TOGETHER_API_KEY)
# - You can change model/provider later without touching node/graph code.
# - If TOGETHER_API_KEY is missing, set it in a .env file like:
#     TOGETHER_API_KEY=your_key_here
#   or export it in your shell before running Streamlit.

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder
from datetime import datetime
import hashlib
import re
from imports import *

load_dotenv()

# Load Together API key from env/.env and warn if missing
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("[WARN] TOGETHER_API_KEY not found in environment (.env). RAG LLM calls will fail until you set it.")

prompt_template = PromptTemplate.from_template(
    """
    You are a helpful assistant. Answer the user's question **using only** the provided context.
    - If the answer is not present in the context, reply: "I don't know." Do not hallucinate.
    - Be concise, clear and polite.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

 # Together model (you can change this centrally)
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# Initializes the LLM (Language Model) from Together AI for generating answers.
# You can change model/temperature/tokens here centrally.
llm = ChatTogether(
    model=TOGETHER_MODEL,
    temperature=0.4,
    max_tokens=1024,
    together_api_key=TOGETHER_API_KEY,
)

rag_chain = prompt_template | llm | StrOutputParser()

def llm_generate(system_prompt: str, user_prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
    """Unified LLM call through Together.
    - If `temperature`/`max_tokens` are provided, build a temporary ChatTogether with overrides.
    - Otherwise reuse the global `llm` instance.
    """
    prompt = f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    local_llm = llm
    if (temperature is not None) or (max_tokens is not None):
        local_llm = ChatTogether(
            model=TOGETHER_MODEL,
            temperature=temperature if temperature is not None else 0.4,
            max_tokens=max_tokens if max_tokens is not None else 1024,
            together_api_key=TOGETHER_API_KEY,
        )
    return local_llm.invoke(prompt)

def get_rag_chain():
    """Expose the RAG chain so Streamlit/app code can pipe context+question easily."""
    return rag_chain


# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
PERSIST_DIR = "chroma_db"               # klasörünle uyumlu
COLLECTION_NAME = "rag_chunks"
SPLIT_CHARS = 1200
SPLIT_OVERLAP = 150
PROJECT_ID = "odine_rag_demo"
TOP_K_INITIAL = 30
TOP_R = 5
RERANK_THRESHOLD = 0.5
# ----------------------------------------

os.makedirs(PERSIST_DIR, exist_ok=True)

# singleton-like service objects
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
_vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=PERSIST_DIR
)
try:
    _reranker = CrossEncoder(RERANK_MODEL)
except Exception as e:
    print("[WARN] Reranker initialization failed:", e)
    _reranker = None

def embeddings() -> HuggingFaceEmbeddings:
    return _embeddings

def vectordb() -> Chroma:
    return _vectordb

def reranker() -> CrossEncoder:
    return _reranker


# --- Usage Notes ---
# - Import services in nodes/graphs like:
#     from services import vectordb, embeddings, reranker, llm_generate, get_rag_chain
# - Set TOGETHER_API_KEY in your .env
# - Vector DB is persisted under PERSIST_DIR; do not recreate per request.

# --- Small helper utilities (exported for imports in graph1) ---

def iso_now() -> str:
  """Return current UTC time in ISO-8601 format."""
  return datetime.utcnow().isoformat()

def short_hash_bytes(data: bytes, length: int = 8) -> str:
  """Short SHA-256 hex digest of raw bytes."""
  return hashlib.sha256(data).hexdigest()[:length]

def short_hash_text(text: str, length: int = 8) -> str:
  """Short SHA-256 hex digest of a UTF-8 string."""
  return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]

def slugify(value: str) -> str:
  """Return a filesystem/URL-friendly slug (lowercase, hyphen-separated)."""
  value = value.lower()
  value = re.sub(r"[^a-z0-9]+", "-", value)
  return value.strip("-")
