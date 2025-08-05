from rag_metric import process_pdf
import streamlit as st

st.title("LangGraph RAG Evaluator")
uploaded_pdf = st.file_uploader("PDF yükle", type="pdf")
question = st.text_input("Soru girin:")

if uploaded_pdf and question:
    with st.spinner("İşleniyor..."):
        # PDF'i geçici dosyaya kaydet
        with open("uploaded.pdf", "wb") as f:
            f.write(uploaded_pdf.read())

        answer, metrics = process_pdf("uploaded.pdf", question)
        st.markdown("### 🧠 Cevap:")
        st.write(answer)
        st.markdown("### 📊 Metrikler:")
        for key, value in metrics.items():
            st.write(f"{key}: {value:.3f}")