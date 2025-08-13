# ================================
# services.py
# Ortak servisler: LLM, prompt, embedding, vektör DB, reranker ve küçük yardımcılar.
# Bu dosyadan graph/node katmanı yalnızca fonksiyon çağırır; config burada merkezidir.
# ================================
import os
# .env dosyasından ortam değişkenlerini (API key vb.) okumak için
from dotenv import load_dotenv
# LangChain prompt ve Together Chat arayüzü
from langchain_core.prompts import PromptTemplate
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser
# Reranker (CE) için Sentence-Transformers CrossEncoder
from sentence_transformers import CrossEncoder
# Küçük yardımcılar: zaman, hash, regex ve ortak importlar
from datetime import datetime
import hashlib
import re
from imports import *

# .env içeriğini process ortamına yükler
load_dotenv()

# Together API anahtarını ortamdan çek; yoksa uyarı bas
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
if not TOGETHER_API_KEY:
    print("[WARN] TOGETHER_API_KEY not found in environment (.env). RAG LLM calls will fail until you set it.")

# RAG cevabı için sıkı şablon: sadece CONTEXT kullan, bilinmiyorsa "I don't know."
# Ayrıca kısa ve sayfa atıflı cevaplar hedeflenir.
prompt_template = PromptTemplate.from_template(
    """
    You are a RAG assistant. Answer the user's question **using only** the CONTEXT.
    Rules:
    - If the answer is not in the context, say exactly: "I don't know." (in English)
    - Be concise. Prefer bullet points when listing items.
    - Do not invent facts or numbers. Do not use outside knowledge.
    - If multiple context chunks disagree, state the uncertainty.

    CONTEXT (verbatim excerpts from the PDF):
    {context}

    QUESTION:
    {question}

    Write the final answer below. If applicable, include short inline citations like (p.{{page}}) using page hints present in the context text.
    ANSWER:
    """
)

# Tek noktadan model seçimi (Together kataloğundan)
TOGETHER_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"

# LLM istemcisi: sıcaklığı düşürerek deterministik davranış; token limiti 1024
llm = ChatTogether(
    model=TOGETHER_MODEL,
    temperature=0.2,
    max_tokens=1024,
    together_api_key=TOGETHER_API_KEY,
)

# Basit zincir: Prompt → LLM → metni çıkar
rag_chain = prompt_template | llm | StrOutputParser()

# Düz sohbet/LLM çağrısı için ortak yardımcı (RAG dışı kullanımda da iş görür)
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

# Dışarıya zinciri veren küçük sarmalayıcı (UI/graph kolay erişsin diye)
def get_rag_chain():
    """Expose the RAG chain so Streamlit/app code can pipe context+question easily."""
    return rag_chain


# ========== RAG CONFIG ==========
# Embedding ve retrieval ayarları; tek yerden değiştir.
# all-mpnet-base-v2 → 768d; Chroma koleksiyonunu buna göre oluştur.
# TOP_K_INITIAL: ilk aday sayısı; TOP_R: LLM'e gidecek üst parçalar
# RERANK_THRESHOLD: düşükse daha fazla parça tutulur; yüksekse agresif filtreler
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
RERANK_MODEL = "BAAI/bge-reranker-base"
PERSIST_DIR = "chroma_db"               # klasörünle uyumlu
COLLECTION_NAME = "rag_chunks"
SPLIT_CHARS = 1200
SPLIT_OVERLAP = 150
PROJECT_ID = "odine_rag_demo"
TOP_K_INITIAL = 40
TOP_R = 6
RERANK_THRESHOLD = 0.35
# ----------------------------------------

# Chroma kalıcılık klasörünü garanti et
os.makedirs(PERSIST_DIR, exist_ok=True)

# Embedding nesnesi (Sentence-Transformers tabanlı)
_embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
# Kalıcı Chroma koleksiyonu; embedding fonksiyonu ile bağlanır
_vectordb = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=PERSIST_DIR
)
# Reranker (CrossEncoder) isteğe bağlı; inemezse None ile devam ederiz (fallback)
try:
    _reranker = CrossEncoder(RERANK_MODEL)
except Exception as e:
    print("[WARN] Reranker initialization failed:", e)
    _reranker = None

# Dış dünyaya tekil embedding nesnesini ver
def embeddings() -> HuggingFaceEmbeddings:
    return _embeddings

# Dış dünyaya tekil Chroma istemcisini ver
def vectordb() -> Chroma:
    return _vectordb

# Dış dünyaya (varsa) CrossEncoder'ı ver; yoksa None döner
def reranker() -> CrossEncoder:
    return _reranker


# Küçük yardımcılar: zaman damgası, kısa hash, basit slug
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
