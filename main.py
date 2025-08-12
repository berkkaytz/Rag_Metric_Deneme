from graph import runnable_pdf, runnable_rag

# Runs the PDF upload LangGraph workflow
def run_pdf_graph(pdf_path: str, question: str):
    initial_state = {
        "pdf_path": pdf_path,
        "question": question
    }
    final_state = runnable_pdf.invoke(initial_state)
    return final_state["answer"], final_state["metrics"]

# Runs the RAG-assisted response generation LangGraph workflow
def run_rag_graph(pdf_path: str, question: str):
    initial_state = {
        "pdf_path": pdf_path,
        "question": question
    }
    final_state = runnable_rag.invoke(initial_state)
    return final_state["answer"], final_state["metrics"]
