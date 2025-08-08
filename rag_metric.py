from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from langchain_core.documents import Document
from bert_score import score
from langgraph.graph import StateGraph
from typing import TypedDict
import evaluate

#Uploads PDF & Adds them to state.
def load_pdf_node(state: dict) -> dict:
    loader = PyPDFLoader(state["pdf_path"])
    pages = loader.load()
    state["pages"] = pages 
    return state

def split_chunks_node(state: dict) -> dict:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(state["pages"])
    state["chunks"] = chunks
    return state

#Creating vektor database.   
def embed_node(state: dict) -> dict:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=state["chunks"],
        embedding=embedding_model,
        collection_name="rag_pdf_demo",
        persist_directory="chroma_db"
    )
    state["vectorstore"] = vectorstore
    return state
   


# Defines the prompt template used to instruct the LLM how to respond using the provided context.
prompt_template = PromptTemplate.from_template("""
You're a helpful assistant. Answer the following question based only on the provided context.

Always respond conversationally, clearly and politely.

Context:
{context}

Question:
{question}

Answer:
""")

# Initializes the LLM (Language Model) from Together AI for generating answers.
llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.4,
    max_tokens=512,
    together_api_key="6a761550941ad644f364b032fac79aa5de4172b8abc73cb771cf8c45fb25e83c"
)

# Chains the prompt, LLM, and output parser together into a RAG-style pipeline.
rag_chain = prompt_template | llm | StrOutputParser()

#Gathers most relevant documents from vektor database.
def retrieval_node(state: dict) -> dict:
    retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state
# With the retrieved context LLM generates answer.
def rag_node(state: dict) -> dict:
    answer = rag_chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    state["answer"] = answer
    return state


#Metric calculation
def evaluate_node(state: dict) -> dict:
    _, _, g = score([state["answer"]], [state["context"]], lang="en", verbose=False)
    _, _, cr = score([state["context"]], [state["answer"]], lang="en", verbose=False)
    _, _, ar = score([state["answer"]], [state["question"]], lang="en", verbose=False)

    state["metrics"] = {
        "groundedness": g[0].item(),
        "context_relevance": cr[0].item(),
        "answer_relevance": ar[0].item()
    }
    return state

def run_graph(pdf_path: str, question: str):
    initial_state = {
        "pdf_path": pdf_path,
        "question": question
    }
    final_state = runnable.invoke(initial_state)
    return final_state["answer"], final_state["metrics"]

#Defining LangGraph flow.
class GraphState(TypedDict):
    
    pdf_path: str
    question: str
    pages: list
    chunks: list
    vectorstore: object
    context: str
    answer: str
    metrics: dict


graph = StateGraph(GraphState)
graph.add_node("load_pdf", load_pdf_node)
graph.add_node("split_chunks", split_chunks_node)
graph.add_node("embed", embed_node)
graph.add_node("retrieval", retrieval_node)
graph.add_node("rag", rag_node)
graph.add_node("evaluate", evaluate_node)

graph.set_entry_point("load_pdf")
graph.add_edge("load_pdf", "split_chunks")
graph.add_edge("split_chunks", "embed")
graph.add_edge("embed", "retrieval")
graph.add_edge("retrieval", "rag")
graph.add_edge("rag", "evaluate")
graph.set_finish_point("evaluate")

runnable = graph.compile()

