from imports import PromptTemplate, ChatTogether, StrOutputParser
import os

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
