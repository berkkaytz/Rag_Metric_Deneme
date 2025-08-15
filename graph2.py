# =============================================================
# graph2.py — RAG Answer Graph (LangGraph)
# Sorgu → retrieve (vektör arama + re-rank) → generate (LLM) → evaluate (metrik)
# Retry kapısı ile metrikler eşik altındaysa tekrar üret.
# Bu dosya sadece Graph‑2 (cevaplama) akışını içerir; ingest Graph‑1’dedir.
# =============================================================
# Ortak tipler/objeler (Document, splitter vb.) ve yardımcılar burada toplanır
from imports import *
# Servis katmanı: vektör DB, reranker, RAG zinciri ve kalite ayarları
from services import (
    TOP_K_INITIAL, TOP_R, RERANK_THRESHOLD,
    vectordb, reranker, get_rag_chain, llm_generate
)

# Tip ipuçları ve LangGraph çekirdeği
from typing import List, Dict, Any, Tuple, TypedDict, Optional, Iterable
from langgraph.graph import StateGraph, END

# Graph‑2’nin durum sözlüğü (state). Akış boyunca bu anahtarlar güncellenir.
# - query: Kullanıcı sorusu
# - doc_id: Hangi PDF’e filtreleneceği (yoksa tüm koleksiyon)
# - retrieved: İlk vektör arama dönüşü (Document listesi)
# - reranked: (Document, skor) listesi (CrossEncoder sonrası)
# - top_docs: LLM’e gidecek (en iyi N) konteks parçaları
# - answer: LLM cevabı (model tarafından üretilen metin)
# - metrics: Ölçüm skorları (0..1)
# - attempt/max_retries/threshold: Retry kontrol parametreleri
class G2State(TypedDict, total=False):
    query: str
    doc_id: Optional[str]
    doc_ids: Optional[List[str]]
    model_name: Optional[str]
    retrieved: List[Document]
    reranked: List[Tuple[Document, float]]
    top_docs: List[Tuple[Document, float]]
    answer: str
    metrics: Dict[str, float]
    attempt: int
    max_retries: int
    threshold: float
    trace: List[str]

# En iyi N dokümanı tek bir CONTEXT metnine çevir (LLM’e verilecek gövde)
def _build_context_string(docs: List[Tuple[Document, float]]) -> str:
    return "\n\n---\n\n".join([doc.page_content for doc, _ in docs])


# Basit lexical benzerlik (fallback olarak kullanılır)
def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    return (len(sa & sb) / len(sa | sb)) if sa and sb else 0.0

# Semantic kıyas için BERTScore; offline ortamda import başarısız olabilir → fallback
try:
    from bert_score import score as _bert_score
except Exception:
    _bert_score = None

# Cevaptan "Sources/Kaynaklar" bölümünü ayıkla; sadece gerçek yanıtı ölçelim
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

# Re‑rank sonrası top‑k konteksi kısa bir metin olarak derle (gürültüyü azaltır)
def _top_context_text_from_top_docs(state: G2State, k: int = 5) -> str:
    top_docs = state.get("top_docs") or []
    if top_docs and isinstance(top_docs[0], tuple):
        docs_only = [d for d, _ in top_docs[:k]]
    else:
        docs_only = top_docs[:k]
    return "\n\n---\n\n".join([d.page_content for d in docs_only]) if docs_only else ""

# BERTScore yoksa Jaccard’a düşen güvenli semantic skorlayıcı
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

# 1) RETRIEVE — Vektör arama + CrossEncoder re‑rank
# - doc_id varsa sadece ilgili PDF’te ara
# - İlk adayları k=TOP_K_INITIAL ile getir, CE ile sırala
# - Eşik altını at, TOP_R kadarını LLM için hazırla
def retrieve_node(state: G2State) -> G2State:
    state.setdefault("trace", []).append("retrieve")
    # Paylaşılan Chroma istemcisini al
    db = vectordb()
    # İsteğe bağlı filtre: tek veya çoklu doküman
    _doc_ids = state.get("doc_ids")
    _doc_id  = state.get("doc_id")
    if _doc_ids:
        filt = {"doc_id": {"$in": _doc_ids}}
    elif _doc_id:
        filt = {"doc_id": _doc_id}
    else:
        filt = None
    # İlk vektör arama (embedding uzayında en yakın k aday)
    retrieved: List[Document] = db.similarity_search(state["query"], k=TOP_K_INITIAL, filter=filt)

    # Re‑rank için (soru, aday_parça) çiftlerini hazırla
    pairs = [(state["query"], d.page_content) for d in retrieved]
    # CrossEncoder puanları (daha ince ayrım)
    if pairs:
        scores = reranker().predict(pairs)
        # Yüksekten düşüğe sırala
        ranked = sorted(zip(retrieved, list(scores)), key=lambda x: x[1], reverse=True)
    else:
        ranked = []

    # Düşük skorluları ele (eşik)
    filtered = [(d, s) for d, s in ranked if s >= RERANK_THRESHOLD]
    # LLM’e gidecek en iyi N parça
    top_docs = filtered[:TOP_R] if len(filtered) >= TOP_R else ranked[:TOP_R]

    # Durumu güncelle
    state["retrieved"] = retrieved
    state["reranked"] = ranked
    state["top_docs"] = top_docs
    return state


# 2) GENERATE — RAG zinciri ile LLM cevabı üret
# - Context boşsa "I don't know." dön
# - Aksi halde prompt → LLM → metin akışı
def generate_node(state: G2State) -> G2State:
    state.setdefault("trace", []).append("generate")
    # Prompt + LLM + parser zinciri
    rag = get_rag_chain(model_name=state.get("model_name"))
    # Seçilen top parçaları tek bir CONTEXT’e çevir
    context = _build_context_string(state.get("top_docs", []))
    # Konteks yoksa modelden cevap isteme
    if not context.strip():
        state["answer"] = "I don't know."
        return state

    # LLM’den cevabı iste ve sadece düz metni sakla
    result = rag.invoke({"context": context, "question": state["query"]})
    state["answer"] = result.content if hasattr(result, "content") else (result if isinstance(result, str) else str(result))
    return state


# 3) EVALUATE — Metrikler (semantic‑first)
# - groundedness: answer ↔ context
# - context_relevance: question ↔ context
# - answer_relevance: answer ↔ question
# Not: Metadata/kaynak listesi ölçüme dahil edilmez (pure answer)
def evaluate_node(state: G2State) -> G2State:
    state.setdefault("trace", []).append("evaluate")
    # Gürültüyü azalt: top‑k kısa konteks derle
    context_compact = _top_context_text_from_top_docs(state, k=5)
    # Fallback: elimizdeki top_docs metni
    if not context_compact:
        context_compact = _build_context_string(state.get("top_docs", []))

    # Sadece gerçek yanıtı ölç (kaynak/metadata hariç)
    answer = _extract_pure_answer(state.get("answer", ""))
    question = state.get("query", "")

    # Semantic groundedness (BERTScore varsa)
    grounded_sem = _safe_bertscore(answer, context_compact)
    # Soru ↔ konteks uyumu
    context_rel  = _safe_bertscore(question, context_compact)
    # Cevap ↔ soru uyumu
    answer_rel   = _safe_bertscore(answer, question)

    # Lexical bonus: en iyi Jaccard eşleşmesini ekle (küçük ağırlıkla)
    ctx = state.get("top_docs", [])
    j_best = max((_jaccard(answer, d.page_content) for d, _ in ctx), default=0.0)
    grounded = 0.8 * grounded_sem + 0.2 * j_best

    # Yuvarla ve [0,1] aralığında tut
    def _rc(x: float) -> float:
        return max(0.0, min(1.0, round(float(x), 3)))

    # Metrikleri state’e yaz
    state["metrics"] = {
        "context_relevance": _rc(context_rel),
        "answer_relevance": _rc(answer_rel),
        "groundedness": _rc(grounded),
    }
    return state

# Retry kapısı — tüm metrikler eşik üstündeyse bitir; değilse ve deneme hakkı varsa tekrar üret
def should_retry(state: G2State) -> str:
    th = state.get("threshold", 0.7)
    mets = state.get("metrics", {})
    ok = all(v >= th for v in mets.values())
    if ok:
        return "finish"
    if state.get("attempt", 0) >= state.get("max_retries", 1):
        return "finish"
    return "retry"


# Bir sonraki deneme için attempt sayacını artır
def retry_gate(state: G2State) -> G2State:
    state.setdefault("trace", []).append("retry_gate")
    state["attempt"] = state.get("attempt", 0) + 1
    return state

# Graph tanımı — düğümleri ekle, kenarları çiz, koşullu geçişleri bağla ve derle
def build_graph2():
    # Tip güvenli state ile bir grafik oluştur
    g = StateGraph(G2State)
    
    g.add_node("retrieve", retrieve_node)
    
    g.add_node("generate", generate_node)
    
    g.add_node("evaluate", evaluate_node)
    
    g.add_node("retry_gate", retry_gate)

    # Giriş düğümü
    g.set_entry_point("retrieve")
    # Akış bağlantıları
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
    # Çalıştırılabilir hâle getir
    return g.compile()


# Varsayılan metrik eşiği (retry mekanizması için)
DEFAULT_THRESHOLD = 0.7

# Dışarıdan çağrılan yardımcı: grafiği kur, input state ver, sonucu UI için paketle
def run_graph2(query: str, doc_id: Optional[str] = None, doc_ids: Optional[List[str]] = None, max_retries: int = 2, threshold: float = DEFAULT_THRESHOLD, model_name: Optional[str] = None) -> Dict[str, Any]:
    # Grafiği derle
    app = build_graph2()
    # Başlangıç state’i ile grafiği çalıştır
    state: G2State = app.invoke({
        "query": query,
        "doc_id": doc_id,
        "doc_ids": doc_ids,
        "attempt": 0,
        "max_retries": max_retries,
        "threshold": threshold,
        "model_name": model_name,
    })

    # UI dostu kaynak listesi hazırla (sayfa, skor, snippet)
    sources: List[Dict[str, Any]] = []
    # Top_docs içinden metadata’yı çıkar ve zenginleştir
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

    # Cevap + metrikler + kaynaklar paketini döndür
    return {
        "answer": state.get("answer", "I don't know."),
        "metrics": state.get("metrics", {}),
        "sources": sources,
        "trace": state.get("trace", []),
    }


# -------------------------------------------------------------
# Streaming execution for Graph-2
# Emits step names as they execute so the UI can update a single line
# Steps: retrieve -> generate -> evaluate -> (retry_gate -> generate -> evaluate)* -> done

def run_graph2_stream(
    query: str,
    doc_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    max_retries: int = 2,
    threshold: float = DEFAULT_THRESHOLD,
    model_name: Optional[str] = None,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (step_name, payload) as Graph-2 advances.

    Yields:
      ("retrieve", {})
      ("generate", {})
      ("evaluate", {"metrics": {...}})
      ("retry_gate", {"attempt": int})  # if retry triggered
      ...
      ("done", {"result": {...}})
    """
    # Initial state
    state: G2State = {
        "query": query,
        "doc_id": doc_id,
        "doc_ids": doc_ids,
        "attempt": 0,
        "max_retries": max_retries,
        "threshold": threshold,
        "model_name": model_name,
        "trace": [],
    }

    # 1) RETRIEVE
    yield ("retrieve", {})
    state = retrieve_node(state)

    # Loop: GENERATE -> EVALUATE (-> RETRY?)
    while True:
        # 2) GENERATE
        yield ("generate", {})
        state = generate_node(state)

        # 3) EVALUATE
        state = evaluate_node(state)
        yield ("evaluate", {"metrics": state.get("metrics", {})})

        # Decide
        decision = should_retry(state)
        if decision == "finish":
            break
        # else retry
        yield ("retry_gate", {"attempt": state.get("attempt", 0) + 1})
        state = retry_gate(state)
        # (loop continues with generate again)

    # Prepare UI-friendly sources
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

    result = {
        "answer": state.get("answer", "I don't know."),
        "metrics": state.get("metrics", {}),
        "sources": sources,
        "trace": state.get("trace", []),
    }

    yield ("done", {"result": result})

# Hızlı yerel test (bu dosyayı doğrudan çalıştırırsan)
if __name__ == "__main__":
    # Örnek bir çağrı (doc_id verilmeden)
    out = run_graph2("Belgenin ana bulguları nelerdir?", doc_id=None)
    # Konsola yazdır
    print("\nAnswer:\n", out["answer"]) 
    print("Metrics:", out["metrics"]) 
    print("Sources:", [(s["page"], round(s["rerank_score"], 3)) for s in out["sources"]])