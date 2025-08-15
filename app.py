# =============================================================
# app.py — Streamlit UI for LangGraph RAG
# PDF yükle → Graph‑1 (ingest) → RAG toggle → Graph‑2 (cevap + metrik + kaynak)
# Bu dosya sadece UI akışını yönetir; iş mantığı graph1/graph2’dedir.
# =============================================================
import os
import hashlib
# Streamlit: web arayüz bileşenleri
import streamlit as st

# Backend fonksiyonları: Graph‑1 (ingest), Graph‑2 (RAG) ve düz LLM çağrısı
from graph1_pdf_upload import run_graph1, run_graph1_stream
from graph2 import run_graph2, run_graph2_stream
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
# --- Session state defaults (ensure keys exist before use) ---
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = False
if "chat" not in st.session_state:
    st.session_state.chat = []
if "model_name" not in st.session_state:
    st.session_state.model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
if "last_pdf_md5" not in st.session_state:
    st.session_state.last_pdf_md5 = None

# Multi-PDF session state
if "doc_index" not in st.session_state:  # md5 -> doc_id
    st.session_state.doc_index = {}
if "docs" not in st.session_state:       # doc_id -> display name
    st.session_state.docs = {}
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None

# Top bar: company logo (left) + app title (right)
logo_path = "download.jpeg"  # place your company logo file in project root
with st.container():
    # Removed logo display here as per instructions
    st.title("🧠 LangGraph RAG Evaluator")

# --- Sidebar: moved logo display here ---
with st.sidebar:
    if os.path.exists(logo_path):
        st.image(logo_path, width=112)
    else:
        st.write("")  # keep spacing if logo not found

    # Sidebar başlığı
    st.header("📄 PDF & RAG")

    # PDF Progress section for Graph-1 live updates
    st.subheader("PDF Progress")
    g1_status_slot = st.empty()

    # RAG progress status area (Graph‑2 live steps)
    st.subheader("RAG Progress")
    g2_status_slot = st.empty()

    # Tek PDF dosyası yükleme alanı
    uploaded_pdfs = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

    # Dosya geldiğinde: diske kaydet + Graph‑1’i çalıştır
    if uploaded_pdfs:
        # At least one file selected — process each if not already ingested
        for uploaded_pdf in uploaded_pdfs:
            file_name = getattr(uploaded_pdf, "name", "uploaded.pdf")
            file_bytes = uploaded_pdf.getvalue()
            current_md5 = hashlib.md5(file_bytes).hexdigest()

            # Skip if identical file already processed this session
            if current_md5 in st.session_state.doc_index:
                already_id = st.session_state.doc_index[current_md5]
                # Removed st.info line as per instructions
                # Ensure bookkeeping
                st.session_state.docs[already_id] = file_name
                continue

            # Save to disk
            temp_path = file_name if file_name.lower().endswith(".pdf") else "uploaded.pdf"
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            # Live node-by-node status in the sidebar PDF Progress area (Graph‑1 stream)
            g1_status_slot.markdown("**Processing PDF (Graph‑1)…**")

            new_doc_id = None
            for step, payload in run_graph1_stream(temp_path):
                if step == "load_pdf":
                    g1_status_slot.markdown(f"**Processing PDF (Graph‑1) — {file_name}**\n- 🔄 Loading file")
                elif step == "split":
                    g1_status_slot.markdown(f"**Processing PDF (Graph‑1) — {file_name}**\n- ✅ Loading file\n- 🔄 Splitting into chunks")
                elif step == "store":
                    g1_status_slot.markdown(f"**Processing PDF (Graph‑1) — {file_name}**\n- ✅ Loading file\n- ✅ Splitting into chunks\n- 🔄 Embedding & storing in vector DB")
                elif step == "done":
                    new_doc_id = payload
                    g1_status_slot.markdown(f"**Processing complete (Graph‑1) — {file_name}.**\n- ✅ Loading file\n- ✅ Splitting into chunks\n- ✅ Embedding & storing in vector DB")

            if new_doc_id:
                # Bookkeeping for multi-doc
                st.session_state.doc_index[current_md5] = new_doc_id
                st.session_state.docs[new_doc_id] = file_name
                st.session_state.last_pdf_md5 = current_md5
                st.success(f"PDF uploaded and vectored ✅ doc_id={new_doc_id}")

        # After loop, mark vectors ready if we have any docs
        st.session_state.vector_ready = len(st.session_state.docs) > 0

        # Default active doc: keep previous if still valid, else pick the last ingested
        if st.session_state.active_doc_id not in st.session_state.docs:
            # last ingested = last value in doc_index mapping
            if st.session_state.doc_index:
                last_doc_id = list(st.session_state.doc_index.values())[-1]
                st.session_state.active_doc_id = last_doc_id

    # Checkbox to search across all uploaded PDFs
    search_all = st.checkbox("Search across all uploaded PDFs", value=False)

    # Choose active document for RAG (if multiple uploaded) only if search_all is False
    if st.session_state.docs and not search_all:
        # options are doc_ids; show file names
        options = list(st.session_state.docs.keys())
        default_index = options.index(st.session_state.active_doc_id) if st.session_state.active_doc_id in options else 0
        chosen = st.selectbox(
            "Active document",
            options,
            index=default_index,
            format_func=lambda _id: st.session_state.docs.get(_id, _id),
        )
        st.session_state.active_doc_id = chosen
        st.session_state.doc_id = chosen  # keep backward compatibility

    # Görsel ayraç
    st.divider()
    st.subheader("Model")
    models = [
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "Qwen/Qwen2.5-7B-Instruct",
        "google/gemma-2-9b-it",
    ]
    st.session_state.model_name = st.selectbox(
        "LLM model selection",
        models,
        index=models.index(st.session_state.model_name) if st.session_state.model_name in models else 0,
    )
    # RAG modunu aç/kapat (cevaplar PDF bağlamına göre olsun mu?)
    st.session_state.rag_mode = st.toggle("RAG Mode", value=st.session_state.rag_mode)
    # Vektör DB hazır değilse uyar (PDF yüklenmedi)
    if st.session_state.rag_mode and not st.session_state.vector_ready:
        st.warning("For RAG you have to upload a PDF.")

# Ana sohbet alanı

# Sohbet geçmişini sırayla çiz (en eski en üstte, en yeni en altta)
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

# Kullanıcıdan yeni mesaj al (input her zaman sayfanın en altında kalır)
user_input = st.chat_input("Type your message…")

# Mesaj gelirse: önce geçmişe ekle, sonra RAG mi düz LLM mi karar ver
if user_input:
    # Kullanıcı mesajını geçmişe yaz
    st.session_state.chat.append({"role": "user", "content": user_input})
    # Show the user message immediately in this run
    with st.chat_message("user"):
        st.markdown(user_input)

    # RAG modu açıkken PDF tabanlı cevap üret
    if st.session_state.rag_mode:
        # PDF/vektör hazır değilse bilgi mesajı gönder
        if not st.session_state.vector_ready:
            assistant_text = "RAG mode is on but no PDF uploaded. Please upload a PDF first."
            st.session_state.chat.append({"role": "assistant", "content": assistant_text})
        else:
            # Sidebar progress slot for RAG steps (created earlier in sidebar)
            status = g2_status_slot
            status.markdown("**RAG (Graph‑2)**\n- ⏳ Starting…")

            final_result = None

            if search_all:
                target_doc_ids = list(st.session_state.docs.keys())
            else:
                target_doc_ids = [st.session_state.active_doc_id]

            for step, payload in run_graph2_stream(
                user_input,
                doc_ids=target_doc_ids,
                max_retries=2,
                model_name=st.session_state.model_name,
            ):
                if step == "retrieve":
                    status.markdown("**RAG (Graph‑2)**\n- 🔄 Retrieving candidates")
                elif step == "generate":
                    status.markdown("**RAG (Graph‑2)**\n- ✅ Retrieved\n- 🔄 Generating answer")
                elif step == "evaluate":
                    mets = payload.get("metrics", {})
                    status.markdown(
                        "**RAG (Graph‑2)**\n"
                        "- ✅ Retrieved\n"
                        "- ✅ Generated\n"
                        f"- 🔄 Evaluating… (ctx={mets.get('context_relevance',0):.2f}, ans={mets.get('answer_relevance',0):.2f}, grd={mets.get('groundedness',0):.2f})"
                    )
                elif step == "retry_gate":
                    attempt = payload.get("attempt", 1)
                    status.markdown(f"**RAG (Graph‑2)**\n- ♻️ Retry #{attempt} (metrics under threshold)")
                elif step == "done":
                    final_result = payload.get("result")
                    status.markdown("**RAG (Graph‑2)**\n- ✅ Retrieved\n- ✅ Generated\n- ✅ Evaluated\n- **Done.**")

            # Final render after stream completes (answer + sources/metrics in chat)
            if final_result:
                with st.chat_message("assistant"):
                    assistant_text = _as_text(final_result.get("answer", "(empty)"))
                    st.markdown(assistant_text)

            sources = final_result.get("sources", [])
            metrics = final_result.get("metrics", {})
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

    

            # Append to history
            meta = {"sources": sources, "metrics": metrics}
            st.session_state.chat.append({"role": "assistant", "content": assistant_text, "meta": meta})
    # RAG kapalıyken: düz LLM sohbeti (services.llm_generate)
    else:
        # Basit sistem mesajı (role)
        system = "You are a helpful assistant."
        # LLM’den yanıt al (Together/Ollama via services)
        with st.chat_message("assistant"):
            with st.spinner("LLM is generating a response…"):
                assistant_text = _as_text(llm_generate(system, user_input, max_tokens=512, model_name=st.session_state.model_name))
            st.markdown(assistant_text)
            # Asistan yanıtını geçmişe ekle
            st.session_state.chat.append({"role": "assistant", "content": assistant_text})

# Bilgilendirme notu: RAG açık/kapalı davranışı
st.caption("When RAG mode is on, answers are generated based on the PDF context and sources/metrics are displayed. When RAG mode is off, plain LLM chat is used.")