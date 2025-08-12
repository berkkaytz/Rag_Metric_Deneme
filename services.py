import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_together import ChatTogether
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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
    together_api_key= os.getenv("TOGETHER_API_KEY")
)

# Chains the prompt, LLM, and output parser together into a RAG-style pipeline.
rag_chain = prompt_template | llm | StrOutputParser()
