import os
import streamlit as st

from graph1_pdf_upload import run_graph1
from graph2 import run_graph2
from services import llm_generate

st.set_page_config(page_title="LangGraph RAG Evaluator", page_icon="🧠", layout="wide")
st.title("🧠 LangGraph RAG Evaluator")

# ---- Session State ----
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
if "vector_ready" not in st.session_state:
    st.session_state.vector_ready = False
if "rag_mode" not in st.session_state:
    st.session_state.rag_mode = False
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {role: "user"|"assistant", content: str, meta: Optional}

# ---- Sidebar: PDF Upload + RAG Toggle ----
with st.sidebar:
    st.header("📄 PDF & RAG")
    uploaded_pdf = st.file_uploader("PDF yükle", type=["pdf"], accept_multiple_files=False)

    if uploaded_pdf is not None:
        # Save to disk
        temp_path = "uploaded.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_pdf.read())
        # Run Graph‑1 (ingest)
        with st.spinner("PDF işleniyor (Graph‑1)…"):
            doc_id = run_graph1(temp_path)
        st.session_state.doc_id = doc_id
        st.session_state.vector_ready = True
        st.success(f"PDF yüklendi ve vektörlendi ✅ doc_id={doc_id}")

    st.divider()
    st.session_state.rag_mode = st.toggle("RAG modu (PDF'e dayalı cevap)", value=st.session_state.rag_mode)
    if st.session_state.rag_mode and not st.session_state.vector_ready:
        st.warning("RAG için önce PDF yüklemelisin.")

# ---- Chat UI ----
chat_container = st.container()
with chat_container:
    # Show existing conversation
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"]) 
            # show meta (sources + metrics) if present
            meta = msg.get("meta")
            if meta and msg["role"] == "assistant":
                sources = meta.get("sources") or []
                metrics = meta.get("metrics") or {}
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

    # Input box
    user_input = st.chat_input("Mesajını yaz…")

    if user_input:
        # append user message
        st.session_state.chat.append({"role": "user", "content": user_input})

        # RAG or plain chat
        if st.session_state.rag_mode:
            if not st.session_state.vector_ready:
                assistant_text = "RAG modu açık ama PDF yok. Lütfen önce PDF yükle."
                st.session_state.chat.append({"role": "assistant", "content": assistant_text})
            else:
                with st.chat_message("assistant"):
                    with st.spinner("RAG (Graph‑2) çalışıyor…"):
                        result = run_graph2(user_input, doc_id=st.session_state.doc_id, max_retries=2)
                    assistant_text = result.get("answer", "(boş)")
                    st.markdown(assistant_text)
                    # render meta (sources + metrics) immediately for this turn
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
                    # keep meta for sources + metrics rendering in history
                    meta = {"sources": result.get("sources", []), "metrics": result.get("metrics", {})}
                    st.session_state.chat.append({"role": "assistant", "content": assistant_text, "meta": meta})
        else:
            # Plain LLM (no RAG) using Together/Ollama via services.llm_generate
            with st.chat_message("assistant"):
                with st.spinner("LLM yanıt üretiyor…"):
                    system = "You are a helpful assistant."
                    assistant_text = llm_generate(system, user_input, max_tokens=512)
                st.markdown(assistant_text)
                st.session_state.chat.append({"role": "assistant", "content": assistant_text})

# (Optional) Footer info
st.caption("RAG açıkken cevaplar PDF bağlamıyla üretilir ve kaynak/metric görüntülenir. RAG kapalıyken düz LLM sohbet edilir.")