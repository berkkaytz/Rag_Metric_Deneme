from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from langchain_core.documents import Document
from bert_score import score
from langgraph.graph import StateGraph, END
from typing import TypedDict
from services import rag_chain
from services import vectordb, embeddings, reranker
from langchain_chroma import Chroma
import os

# Global chunking configuration (used by Graph-1 and Graph-2)
SPLIT_CHARS = 1000
SPLIT_OVERLAP = 200