from imports import StateGraph, TypedDict
from imports import (
    load_pdf_node,
    split_chunks_node,
    embed_node,
    retrieval_node,
    rag_node,
    evaluate_node,
)
# Defining LangGraph flow.
class GraphState(TypedDict):

    pdf_path: str
    question: str
    pages: list
    chunks: list
    vectorstore: object
    context: str
    answer: str
    metrics: dict



# PDF upload, split, embed flow
graph_pdf = StateGraph(GraphState)
graph_pdf.add_node("load_pdf", load_pdf_node)
graph_pdf.add_node("split_chunks", split_chunks_node)
graph_pdf.add_node("embed", embed_node)

graph_pdf.set_entry_point("load_pdf")
graph_pdf.add_edge("load_pdf", "split_chunks")
graph_pdf.add_edge("split_chunks", "embed")
graph_pdf.set_finish_point("embed")

runnable_pdf = graph_pdf.compile()

# Retrieval, RAG, evaluate flow
graph_rag = StateGraph(GraphState)
graph_rag.add_node("retrieval", retrieval_node)
graph_rag.add_node("rag", rag_node)
graph_rag.add_node("evaluate", evaluate_node)

graph_rag.set_entry_point("retrieval")
graph_rag.add_edge("retrieval", "rag")
graph_rag.add_edge("rag", "evaluate")
graph_rag.set_finish_point("evaluate")

runnable_rag = graph_rag.compile()
