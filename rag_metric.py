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

    # Loads the PDF file, splitting it into separate Document objects for each page.
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()  # her sayfa bir Document nesnesi


    # Splits the documents into overlapping chunks to preserve context across boundaries.
    # Here, each chunk is 1000 characters with a 200-character overlap.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)

    
    # Initializes the embedding model to convert text chunks into vector representations.
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # Stores the embedded chunks into a Chroma vector database for retrieval.
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

def ask_question(vectorstore, question: str):
    
    # Converts the vectorstore into a retriever to fetch relevant documents based on the question.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs: list[Document] = retriever.invoke(question)

    # Combines the retrieved documents into a single context string.
    context = "\n\n".join([doc.page_content for doc in docs])

    # Uses the RAG chain to generate an answer from the question and retrieved context.
    answer = rag_chain.invoke({
        "context": context,
        "question": question
    })

    print("🧠 Cevap:", answer)
    return question, context, answer

# Imports the BERTScore metric for evaluating answer quality.
from bert_score import score

def evaluate_metrics(question: str, context: str, answer: str):
    
    # Measures groundedness: how well the answer is supported by the retrieved context.
    _, _, groundedness_f1 = score([answer], [context], lang="en", verbose=False)
    
    # Measures context relevance: how well the context covers the answer.
    _, _, context_rel_f1 = score([context], [answer], lang="en", verbose=False)
    
    # Measures answer relevance: how well the answer relates to the original question.
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
