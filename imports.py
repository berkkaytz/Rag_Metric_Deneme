from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from langchain_core.documents import Document
from bert_score import score
from langgraph.graph import StateGraph, END
from typing import TypedDict
from services import rag_chain
from services import vectordb, embeddings, reranker
from langchain_chroma import Chroma

# Global chunking configuration (used by Graph-1 and Graph-2)
SPLIT_CHARS = 1000
SPLIT_OVERLAP = 200

def load_pdf_node(state: dict) -> dict:
    loader = PyPDFLoader(state["pdf_path"])
    pages = loader.load()
    state["pages"] = pages 
    return state

def split_chunks_node(state: dict) -> dict:
    splitter = RecursiveCharacterTextSplitter(chunk_size=SPLIT_CHARS, chunk_overlap=SPLIT_OVERLAP)
    chunks = splitter.split_documents(state["pages"])
    state["chunks"] = chunks
    return state

# Creating vektor database.   
def embed_node(state: dict) -> dict:
    # Persist chunks into the shared vector DB (idempotent per doc if you want)
    db = vectordb()
    db.add_documents(state["chunks"])  # assumes chunks already carry metadata
    db.persist()
    state["vectorstore"] = db
    return state

# Gathers most relevant documents from vektor database.
def retrieval_node(state: dict) -> dict:
    db = vectordb()
    filt = {"doc_id": state.get("doc_id")} if state.get("doc_id") else None
    docs = db.similarity_search(state["question"], k=10, filter=filt)
    state["candidates"] = docs
    return state

def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

def rerank_node(state: dict) -> dict:
    query = state["question"]
    docs = state.get("candidates", [])
    if not docs:
        state["reranked"] = []
        state["context"] = ""
        return state

    try:
        ce = reranker()
        pairs = [(query, d.page_content) for d in docs]
        scores = ce.predict(pairs)
        ranked = sorted(zip(docs, list(scores)), key=lambda x: x[1], reverse=True)
    except Exception:
        # Fallback: simple lexical similarity
        ranked = sorted(
            [(d, _jaccard(query, d.page_content)) for d in docs],
            key=lambda x: x[1], reverse=True
        )

    state["reranked"] = ranked
    # Build context from top 5 chunks
    top_docs = [d for d, _ in ranked[:5]]
    state["context"] = "\n\n".join([d.page_content for d in top_docs])
    return state

# Metric calculation
def evaluate_node(state: dict) -> dict:
    _, _, g = score([state["answer"]], [state["context"]], lang="en", verbose=False)
    _, _, cr = score([state["context"]], [state["answer"]], lang="en", verbose=False)
    _, _, ar = score([state["answer"]], [state["question"]], lang="en", verbose=False)

    state["metrics"] = {
        "groundedness": g[0].item(),
        "context_relevance": cr[0].item(),
        "answer_relevance": ar[0].item()
    }
    return state


def rag_node(state: dict) -> dict:
    answer = rag_chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    state["answer"] = answer
    return state