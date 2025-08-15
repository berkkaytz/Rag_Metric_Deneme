# =============================================================
# graph1_pdf_upload.py — Ingest Graph (LangGraph)
# PDF → load → split (chunk) → enrich metadata → store in vector DB (Chroma)
# Bu dosya sadece ingest (Graph‑1) işini yapar; cevaplama Graph‑2’dedir.
# =============================================================
# Ortak tipler/araçlar (Document, PyPDFLoader, splitter vb.)
from imports import *
# Yol/çevre değişkenleri ve dosya adı işlemleri için
import os

# Yardımcılar: dosya hash’i, metin hash’i, zaman damgası, embedding modeli adı, slugify
from services import short_hash_bytes, short_hash_text, iso_now, EMBED_MODEL, slugify

# Varsayılan proje etiketi (metadata’da tutulur)
PROJECT_ID = os.getenv("PROJECT_ID", "default_project")

# Tip ipuçları ve LangGraph çekirdeği
from typing import List, TypedDict, Iterable, Tuple
from langgraph.graph import StateGraph, END


# Graph‑1 durum sözlüğü:
# - pdf_path: Yüklü PDF’in yolu (geçici dosya)
# - documents: PDF’ten sayfa bazlı Document listesi
# - chunks: Parçalanmış (chunk’lanmış) Document listesi
# - doc_id: Belgeyi tanımlayan deterministik kimlik (slug+hash)
# - result: Upsert sonucu (log/özet)
# - trace: İşlem izlerini tutan liste
class G1State(TypedDict, total=False):
    pdf_path: str
    documents: List[Document]
    chunks: List[Document]
    doc_id: str
    result: str
    trace: List[str]

# 1) LOAD — PDF’i oku ve deterministik doc_id oluştur
def load_pdf_node(state: G1State) -> G1State:
    # Debug: girişte hangi dosya geldiğini göster
    print("[Graph1] load_pdf_node: path=", state.get("pdf_path"))
    # PDF loader’ı hazırla
    loader = PyPDFLoader(state["pdf_path"])
    # Sayfa bazlı Document listesi al
    docs = loader.load()
    # Dosya baytlarından kısa hash üret (doc_id tutarlılığı için)
    with open(state["pdf_path"], "rb") as f:
        fh = short_hash_bytes(f.read())
    # Dosya adına slug + içerik hash’i → doc_id
    state["doc_id"] = f"{slugify(os.path.basename(state['pdf_path']))}-{fh}"
    # Okunan sayfaları state’e yaz
    state["documents"] = docs
    # Kaç sayfa/Document yüklendiğini logla
    print("[Graph1] load_pdf_node: docs=", len(docs))
    # Trace güncelle
    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("load_pdf")
    return state


# 2) SPLIT — Belgede eksik varsa yükü tazele, sonra chunk’la ve metadata zenginleştir
def split_node(state: G1State) -> G1State:
    # Güvenlik: documents yoksa yeniden yükle
    if "documents" not in state or state["documents"] is None:
        print("[Graph1] split_node: documents missing, reloading")
        loader = PyPDFLoader(state["pdf_path"])
        state["documents"] = loader.load()
    # Güvenlik: doc_id yoksa aynı yöntemle baştan üret
    if "doc_id" not in state or not state["doc_id"]:
        with open(state["pdf_path"], "rb") as f:
            fh = short_hash_bytes(f.read())
        state["doc_id"] = f"{slugify(os.path.basename(state['pdf_path']))}-{fh}"
        print("[Graph1] split_node: regenerated doc_id=", state["doc_id"])
    # Chunk’lama ayarları (karakter bazlı; overlap ile bağlam korunur)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SPLIT_CHARS, chunk_overlap=SPLIT_OVERLAP
    )
    # Sayfaları parçalara böl
    state["chunks"] = splitter.split_documents(state["documents"])
    # Üretilen chunk sayısını logla
    print("[Graph1] split_node: chunks=", len(state["chunks"]))
    # Çevreden proje etiketi al (globals fallback ile NameError önle)
    proj_id = os.getenv("PROJECT_ID", PROJECT_ID if 'PROJECT_ID' in globals() else "default_project")
    # Her chunk için zengin metadata ekle
    for idx, ch in enumerate(state["chunks"]):
        # Kaynak sayfa numarası
        page = int(ch.metadata.get("page") or 0)
        # İçerik metni (token sayımı ve hash için)
        ch_text = ch.page_content
        # Metadata alanları: izlenebilirlik ve kalite/analiz için
        ch.metadata.update({
            "doc_id": state["doc_id"],
            "chunk_id": f"{state['doc_id']}_p{page}_c{idx}",
            "page": page,
            "chunk_index": idx,
            "source": state["pdf_path"],
            "created_at": iso_now(),
            "split_method": "recursive",
            "overlap": SPLIT_OVERLAP,
            "token_count": len(ch_text.split()),
            "hash": short_hash_text(ch_text),
            "embedding_model": EMBED_MODEL,
            "project_id": proj_id,
        })
    # Trace güncelle
    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("split")
    return state


# 3) STORE — Vektör veritabanına idempotent upsert ve kaydet
def enrich_and_store_node(state: G1State) -> G1State:
    # Paylaşılan Chroma istemcisini al
    db = vectordb()
    # Aynı doc_id varsa sil (temiz upsert için)
    try:
        db.delete(where={"doc_id": state["doc_id"]})
    except Exception:
        pass
    # Chunk’ları upsert et
    db.add_documents(state["chunks"])
    # Kalıcı depoya yaz
    db.persist()

    # Özet sonucu state’e yaz (UI/CLI log’u için)
    state["result"] = f"upserted={len(state['chunks'])} doc_id={state['doc_id']}"
    # Trace güncelle
    if "trace" not in state or state["trace"] is None:
        state["trace"] = []
    state["trace"].append("store")
    return state

# Graph tanımı — düğümleri ekle, kenarları bağla ve derle
def build_graph1():
    # Tip güvenli state ile bir grafik oluştur
    g = StateGraph(G1State)
    # Düğüm ekle (load → split → store)
    g.add_node("load_pdf", load_pdf_node)
    # Düğüm ekle (load → split → store)
    g.add_node("split", split_node)
    # Düğüm ekle (load → split → store)
    g.add_node("store", enrich_and_store_node)
    # Giriş düğümü
    g.set_entry_point("load_pdf")
    # Akış bağlantıları
    g.add_edge("load_pdf", "split")
    # Akış bağlantıları
    g.add_edge("split", "store")
    # Akış bağlantıları
    g.add_edge("store", END)
    # Çalıştırılabilir hâle getir
    return g.compile()

# Dışarıdan çağrılan yardımcı: grafiği kur, PDF yolunu ver, doc_id döndür
def run_graph1(pdf_path: str) -> str:
    # Grafiği derle
    app = build_graph1()
    # Başlangıç state ile grafiği çalıştır
    out: G1State = app.invoke({"pdf_path": pdf_path})
    # İlerleme/özet bilgisini logla
    print("[Graph1]", out.get("result"))
    # UI/Graph‑2 için kullanılacak doc_id ve trace’yi döndür
    return out["doc_id"], out.get("trace", [])

# Streaming helper: step-by-step execution that yields current node name
# Yields: "load_pdf" → "split" → "store" and finally ("done", doc_id, trace)

def run_graph1_stream(pdf_path: str) -> Iterable[Tuple[str, str | list]]:
    state: G1State = {"pdf_path": pdf_path, "trace": []}

    # 1) LOAD
    yield ("load_pdf", "")
    state = load_pdf_node(state)

    # 2) SPLIT
    yield ("split", "")
    state = split_node(state)

    # 3) STORE
    yield ("store", "")
    state = enrich_and_store_node(state)

    # DONE
    yield ("done", state.get("doc_id", ""))
    # If UI needs the full trace afterward, it can be read from state or from run_graph1(...)

# Hızlı yerel test (doğrudan çalıştırma)
if __name__ == "__main__":
    # Örnek PDF ile deneyin
    run_graph1("uploaded.pdf")