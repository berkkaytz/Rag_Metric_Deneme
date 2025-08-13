# graph2.py (LangGraph version)
from imports import *
from services import (
    TOP_K_INITIAL, TOP_R, RERANK_THRESHOLD,
    vectordb, reranker, get_rag_chain, llm_generate
)

from typing import List, Dict, Any, Tuple, TypedDict, Optional
from langgraph.graph import StateGraph, END

# ---------------- State ----------------
class G2State(TypedDict, total=False):
    query: str
    doc_id: Optional[str]
    retrieved: List[Document]
    reranked: List[Tuple[Document, float]]
    top_docs: List[Tuple[Document, float]]
    answer: str
    metrics: Dict[str, float]
    attempt: int
    max_retries: int
    threshold: float

# ---------------- Helpers ----------------

def _build_context_string(docs: List[Tuple[Document, float]]) -> str:
    return "\n\n---\n\n".join([doc.page_content for doc, _ in docs])


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

# --- Semantic eval helpers (optional, with safe fallback) ---
try:
    from bert_score import score as _bert_score
except Exception:
    _bert_score = None

def _extract_pure_answer(text: str) -> str:
    if not text:
        return ""
    cuts = [
        "🔗 Sources", "Sources / Metadata", "Kaynaklar", "Kaynak alıntıları",
        "Sources:", "Metadata:", "References"
    ]
    for c in cuts:
        idx = text.find(c)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()

def _top_context_text_from_top_docs(state: G2State, k: int = 5) -> str:
    top_docs = state.get("top_docs") or []
    if top_docs and isinstance(top_docs[0], tuple):
        docs_only = [d for d, _ in top_docs[:k]]
    else:
        docs_only = top_docs[:k]
    return "\n\n---\n\n".join([d.page_content for d in docs_only]) if docs_only else ""

def _safe_bertscore(a: str, b: str) -> float:
    """Return a semantic similarity in [0,1]; falls back to Jaccard if offline."""
    if _bert_score is None:
        return _jaccard(a, b)
    try:
        _, _, f1 = _bert_score([a], [b], lang="en", verbose=False)
        val = float(f1[0])
        return max(0.0, min(1.0, val))
    except Exception:
        return _jaccard(a, b)

# ---------------- Nodes ----------------

def retrieve_node(state: G2State) -> G2State:
    db = vectordb()
    filt = {"doc_id": state.get("doc_id")} if state.get("doc_id") else None
    retrieved: List[Document] = db.similarity_search(state["query"], k=TOP_K_INITIAL, filter=filt)

    # re-rank with cross-encoder
    pairs = [(state["query"], d.page_content) for d in retrieved]
    if pairs:
        scores = reranker().predict(pairs)
        ranked = sorted(zip(retrieved, list(scores)), key=lambda x: x[1], reverse=True)
    else:
        ranked = []

    filtered = [(d, s) for d, s in ranked if s >= RERANK_THRESHOLD]
    top_docs = filtered[:TOP_R] if len(filtered) >= TOP_R else ranked[:TOP_R]

    state["retrieved"] = retrieved
    state["reranked"] = ranked
    state["top_docs"] = top_docs
    return state


def generate_node(state: G2State) -> G2State:
    # Use the configured RAG chain (prompt -> LLM -> string)
    rag = get_rag_chain()
    context = _build_context_string(state.get("top_docs", []))
    if not context.strip():
        state["answer"] = "I don't know."
        return state

    state["answer"] = rag.invoke({"context": context, "question": state["query"]})
    return state


def evaluate_node(state: G2State) -> G2State:
    # 1) Build compact context from top-k reranked docs
    context_compact = _top_context_text_from_top_docs(state, k=5)
    if not context_compact:
        # fallback to whatever context string exists
        context_compact = _build_context_string(state.get("top_docs", []))

    # 2) Extract the pure answer text (exclude Sources/Metadata section)
    answer = _extract_pure_answer(state.get("answer", ""))
    question = state.get("query", "")

    # 3) Compute metrics (semantic-first with lexical fallback)
    grounded_sem = _safe_bertscore(answer, context_compact)
    context_rel  = _safe_bertscore(question, context_compact)
    answer_rel   = _safe_bertscore(answer, question)

    # Add a small lexical bonus for groundedness using best Jaccard match
    ctx = state.get("top_docs", [])
    j_best = max((_jaccard(answer, d.page_content) for d, _ in ctx), default=0.0)
    grounded = 0.8 * grounded_sem + 0.2 * j_best

    # 4) Round & clamp
    def _rc(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 3)))

    state["metrics"] = {
        "context_relevance": _rc(context_rel),
        "answer_relevance": _rc(answer_rel),
        "groundedness": _rc(grounded),
    }
    return state

# ------------- Conditional routing -------------

def should_retry(state: G2State) -> str:
    th = state.get("threshold", 0.7)
    mets = state.get("metrics", {})
    ok = all(v >= th for v in mets.values())
    if ok:
        return "finish"
    if state.get("attempt", 0) >= state.get("max_retries", 1):
        return "finish"
    return "retry"


def retry_gate(state: G2State) -> G2State:
    state["attempt"] = state.get("attempt", 0) + 1
    return state

# ---------------- Build & Run ----------------

def build_graph2():
    g = StateGraph(G2State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("generate", generate_node)
    g.add_node("evaluate", evaluate_node)
    g.add_node("retry_gate", retry_gate)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "evaluate")

    g.add_conditional_edges(
        "evaluate",
        should_retry,
        {
            "retry": "retry_gate",
            "finish": END,
        },
    )
    g.add_edge("retry_gate", "generate")
    return g.compile()


DEFAULT_THRESHOLD = 0.7

def run_graph2(query: str, doc_id: Optional[str] = None, max_retries: int = 2, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    app = build_graph2()
    state: G2State = app.invoke({
        "query": query,
        "doc_id": doc_id,
        "attempt": 0,
        "max_retries": max_retries,
        "threshold": threshold,
    })

    # Prepare UI-friendly sources list
    sources: List[Dict[str, Any]] = []
    for d, score in (state.get("top_docs") or []):
        meta = d.metadata or {}
        sources.append({
            "doc_id": meta.get("doc_id"),
            "chunk_id": meta.get("chunk_id"),
            "page": meta.get("page"),
            "source": meta.get("source"),
            "section": meta.get("section"),
            "title": meta.get("title"),
            "rerank_score": float(score),
            "snippet": d.page_content[:250],
        })

    return {
        "answer": state.get("answer", "I don't know."),
        "metrics": state.get("metrics", {}),
        "sources": sources,
    }


if __name__ == "__main__":
    out = run_graph2("Belgenin ana bulguları nelerdir?", doc_id=None)
    print("\nAnswer:\n", out["answer"]) 
    print("Metrics:", out["metrics"]) 
    print("Sources:", [(s["page"], round(s["rerank_score"], 3)) for s in out["sources"]])