# =============================================================
# app.py — Streamlit UI for LangGraph RAG
# PDF yükle → Graph‑1 (ingest) → RAG toggle → Graph‑2 (cevap + metrik + kaynak)
# Bu dosya sadece UI akışını yönetir; iş mantığı graph1/graph2’dedir.
# =============================================================
import os
# Streamlit: web arayüz bileşenleri
import streamlit as st

# Backend fonksiyonları: Graph‑1 (ingest), Graph‑2 (RAG) ve düz LLM çağrısı
from graph1_pdf_upload import run_graph1
from graph2 import run_graph2
from services import llm_generate

# --- Helper: her türlü LLM çıktısını düz metne çevir ---
def _as_text(x) -> str:
    """AIMessage / dict / repr('content="..." additional_kwargs=...') → plain text"""
    # 1) AIMessage benzeri obje
    if hasattr(x, "content"):
        try:
            return x.content
        except Exception:
            pass
    # 2) dict formatı
    if isinstance(x, dict) and "content" in x:
        try:
            return str(x.get("content", ""))
        except Exception:
            pass
    # 3) str'e çevir ve repr kalıbından içeriği ayıkla
    s = str(x)
    try:
        import re
        m = re.search(r"content=([\\\'\"])(.*?)\1\s+additional_kwargs=", s, re.DOTALL)
        if m:
            return m.group(2)
    except Exception:
        pass
    return s

# Sayfa başlığı/ikon ve geniş layout ayarı
st.set_page_config(page_title="LangGraph RAG Evaluator", page_icon="🧠", layout="wide")
# Uygulama başlığı
st.title("🧠 LangGraph RAG Evaluator")

# ---- Session State ----
# Chat ve çalışma durumu burada tutulur; tarayıcı yenilense bile korunur
# Varsayılan durum anahtarı oluştur
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
# Varsayılan durum anahtarı oluştur
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
# Varsayılan durum anahtarı oluştur
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = False
# Varsayılan durum anahtarı oluştur
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {role: "user"|"assistant", content: str, meta: Optional}

# --- Sidebar: PDF yükleme ve RAG anahtarı ---
with st.sidebar:
    # Sidebar başlığı
    st.header("📄 PDF & RAG")
    # Tek PDF dosyası yükleme alanı
    uploaded_pdf = st.file_uploader("PDF yükle", type=["pdf"], accept_multiple_files=False)

    # Dosya geldiğinde: diske kaydet + Graph‑1’i çalıştır
    if uploaded_pdf is not None:
        # Geçici isimle kaydet (tek dosya iş akışı için yeterli)
        temp_path = "uploaded.pdf"
        # PDF içeriğini yerel dosyaya yaz
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())
        # Ingest akışını gösterge ile çalıştır
        with st.spinner("PDF işleniyor (Graph‑1)…"):
            # Graph‑1: PDF → chunk → vektör DB; geri doc_id döner
            doc_id = run_graph1(temp_path)
        # UI durumunu güncelle (RAG için gerekli)
        st.session_state.doc_id = doc_id
        st.session_state.vector_ready = True
        # Kullanıcıya başarı mesajı
        st.success(f"PDF yüklendi ve vektörlendi ✅ doc_id={doc_id}")

    # Görsel ayraç
    st.divider()
    # RAG modunu aç/kapat (cevaplar PDF bağlamına göre olsun mu?)
    st.session_state.rag_mode = st.toggle("RAG modu (PDF'e dayalı cevap)", value=st.session_state.rag_mode)
    # Vektör DB hazır değilse uyar (PDF yüklenmedi)
    if st.session_state.rag_mode and not st.session_state.vector_ready:
        st.warning("RAG için önce PDF yüklemelisin.")

# Ana sohbet alanı
chat_container = st.container()
# Sohbet geçmişini ve yeni mesajları bu container içinde göster
with chat_container:
    # Önceki konuşmaları sırayla çiz
    for msg in st.session_state.chat:
        # Mesaj balonu (user/assistant)
        with st.chat_message(msg["role"]):
            # Mesaj metnini yazdır (sadece düz metin; ham obje ise ayıkla)
            _content = _as_text(msg.get("content"))
            st.markdown(_content)
            # (Varsa) bu mesaja ait kaynaklar ve metrikler
            meta = msg.get("meta")
            if meta and msg["role"] == "assistant":
                # Kaynaklar: sayfa/başlık/snippet ve re‑rank skoru
                sources = meta.get("sources") or []
                metrics = meta.get("metrics") or {}
                if sources:
                    with st.expander("🔗 Sources / Metadata"):
                        # Her kaynak için ayrıntı
                        for s in sources:
                            st.markdown(
                                f"- **page**: {s.get('page')} | **chunk**: `{s.get('chunk_id')}` | **score**: {s.get('rerank_score'):.3f}\n\n"
                                f"  **title/section**: {s.get('title') or s.get('section') or '-'}\n\n"
                                f"  _{s.get('snippet', '')}_\n"
                            )
                # Metrikler: context/answer relevance ve groundedness
                if metrics:
                    # 3 sütunda metrik gösterimi
                    cols = st.columns(3)
                    keys = ["context_relevance", "answer_relevance", "groundedness"]
                    labels = ["Context Relevance", "Answer Relevance", "Groundedness"]
                    for i, k in enumerate(keys):
                        with cols[i]:
                            st.metric(labels[i], f"{metrics.get(k, 0.0):.3f}")

    # Kullanıcıdan yeni mesaj al
    user_input = st.chat_input("Mesajını yaz…")

    # Mesaj gelirse: önce geçmişe ekle, sonra RAG mi düz LLM mi karar ver
    if user_input:
        # Kullanıcı mesajını geçmişe yaz
        st.session_state.chat.append({"role": "user", "content": user_input})

        # RAG modu açıkken PDF tabanlı cevap üret
        if st.session_state.rag_mode:
            # PDF/vektör hazır değilse bilgi mesajı gönder
            if not st.session_state.vector_ready:
                assistant_text = "RAG modu açık ama PDF yok. Lütfen önce PDF yükle."
                st.session_state.chat.append({"role": "assistant", "content": assistant_text})
            else:
                # Asistan balonu: RAG cevabı ve meta gösterimi
                with st.chat_message("assistant"):
                    # Graph‑2: retrieve → generate → evaluate (+ retry)
                    with st.spinner("RAG (Graph‑2) çalışıyor…"):
                        result = run_graph2(user_input, doc_id=st.session_state.doc_id, max_retries=2)
                    # Modele ait yanıt metni
                    assistant_text = _as_text(result.get("answer", "(boş)"))
                    st.markdown(assistant_text)
                    # Bu tura ait kaynak ve metrikleri anında göster
                    sources = result.get("sources", [])
                    metrics = result.get("metrics", {})
                    if sources:
                        with st.expander("🔗 Sources / Metadata"):
                            for s in sources:
                                st.markdown(
                                    f"- **page**: {s.get('page')} | **chunk**: `{s.get('chunk_id')}` | **score**: {s.get('rerank_score'):.3f}\n\n"
                                    f"  **title/section**: {s.get('title') or s.get('section') or '-'}\n\n"
                                    f"  _{s.get('snippet', '')}_\n"
                                )
                    if metrics:
                        cols = st.columns(3)
                        keys = ["context_relevance", "answer_relevance", "groundedness"]
                        labels = ["Context Relevance", "Answer Relevance", "Groundedness"]
                        for i, k in enumerate(keys):
                            with cols[i]:
                                st.metric(labels[i], f"{metrics.get(k, 0.0):.3f}")
                    # Geçmişte de aynı tur için kaynak/metrik sakla
                    meta = {"sources": result.get("sources", []), "metrics": result.get("metrics", {})}
                    st.session_state.chat.append({"role": "assistant", "content": assistant_text, "meta": meta})
        # RAG kapalıyken: düz LLM sohbeti (services.llm_generate)
        else:
            # Basit sistem mesajı (role)
            system = "You are a helpful assistant."
            # LLM’den yanıt al (Together/Ollama via services)
            with st.chat_message("assistant"):
                with st.spinner("LLM yanıt üretiyor…"):
                    assistant_text = _as_text(llm_generate(system, user_input, max_tokens=512))
                st.markdown(assistant_text)
                # Asistan yanıtını geçmişe ekle
                st.session_state.chat.append({"role": "assistant", "content": assistant_text})

# Bilgilendirme notu: RAG açık/kapalı davranışı
st.caption("RAG açıkken cevaplar PDF bağlamıyla üretilir ve kaynak/metric görüntülenir. RAG kapalıyken düz LLM sohbet edilir.")