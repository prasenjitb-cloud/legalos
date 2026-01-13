from dotenv import load_dotenv
load_dotenv()
from langchain_ollama import ChatOllama
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ---- embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("jurisdesk")  # must exist
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="state_acts"  # same namespace you used
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


# prompt used for the RAG system
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a legal assistant.

Answer ONLY using the provided legal documents.
If the answer is not present, say "Not found in the documents".
Mention the Act name and Section number if available.

Context:
{context}

Question:
{question}

Answer:
"""
)


llm = ChatOllama(
    model="qwen2.5:3b-instruct",
    temperature=0,
)


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("jurisdesk")


q ="what happens if someone murders anyone"

retrieved_docs=retriever.invoke(q)
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.invoke({"context": context_text, "question": q})

answer = llm.invoke(final_prompt)
print(answer.content) 