from main import run_graph
import streamlit as st

st.title("LangGraph RAG Evaluator")
uploaded_pdf = st.file_uploader("PDF yükle", type="pdf")
question = st.text_input("Soru girin:")

if uploaded_pdf and question:
    with st.spinner("İşleniyor..."):
        # PDF'i geçici dosyaya kaydet
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        answer, metrics = run_graph("uploaded.pdf", question)
        st.markdown("### 🧠 Cevap:")
        st.write(answer)
        st.markdown("### 📊 Metrikler:")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.3f}")