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
from services import rag_chain

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

# Creating vektor database.   
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

# Gathers most relevant documents from vektor database.
def retrieval_node(state: dict) -> dict:
    retriever = state["vectorstore"].as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    state["context"] = context
    return state



# Metric calculation
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


def rag_node(state: dict) -> dict:
    answer = rag_chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })
    state["answer"] = answer
    return state