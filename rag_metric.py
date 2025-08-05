from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_together import ChatTogether
from langchain_core.documents import Document
import evaluate

def process_pdf(pdf_path: str, question: str):

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # her sayfa bir Document nesnesi


    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="rag_pdf_demo",
        persist_directory="chroma_db"
    )

    # return vectorstore
    question, context, answer = ask_question(vectorstore, question)
    metrics = evaluate_metrics(question, context, answer)
    return answer, metrics




# Prompt tanımı
prompt_template = PromptTemplate.from_template("""
You're a helpful assistant. Answer the following question based only on the provided context.

Always respond conversationally, clearly and politely.

Context:
{context}

Question:
{question}

Answer:
""")

llm = ChatTogether(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0.4,
    max_tokens=512,
    together_api_key="6a761550941ad644f364b032fac79aa5de4172b8abc73cb771cf8c45fb25e83c"
)

rag_chain = prompt_template | llm | StrOutputParser()

def ask_question(vectorstore, question: str):
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs: list[Document] = retriever.invoke(question)

    
    context = "\n\n".join([doc.page_content for doc in docs])

   
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("🧠 Cevap:", answer)
    return question, context, answer

from bert_score import score

def evaluate_metrics(question: str, context: str, answer: str):
    
    _, _, groundedness_f1 = score([answer], [context], lang="en", verbose=False)
    
    
    _, _, context_rel_f1 = score([context], [answer], lang="en", verbose=False)
    
    
    _, _, answer_rel_f1 = score([answer], [question], lang="en", verbose=False)
    
    print("📊 Metrikler:")
    print(f"groundedness: {groundedness_f1[0]:.3f}")
    print(f"context_relevance: {context_rel_f1[0]:.3f}")
    print(f"answer_relevance: {answer_rel_f1[0]:.3f}")
    
    return {
        "groundedness": groundedness_f1[0].item(),
        "context_relevance": context_rel_f1[0].item(),
        "answer_relevance": answer_rel_f1[0].item()
    }

