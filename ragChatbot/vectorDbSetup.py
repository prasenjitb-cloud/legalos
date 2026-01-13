from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

pc.create_index(
    name="jurisdesk",
    dimension=384,          # bge-small-en = 384
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)



from bs4 import BeautifulSoup
import requests
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re
from langchain_core.documents import Document
from pinecone import Pinecone
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore


FirstLink = "https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle&rpp=845"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.indiacode.nic.in/"
}

session = requests.Session()
html = session.get(FirstLink, headers=headers).text

soup = BeautifulSoup(html)

# print(soup.prettify())


FAILED_LOG = "failed_pdfs.txt"


# ---- embeddings ----
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ---- pinecone client ----
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index("jurisdesk")  # must exist

# ---- vector store ----
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    namespace="state_acts",
)



firstAs= soup.select("table tr td a")

baseUrl= "https://www.indiacode.nic.in"
seen_urls = set()
pdf_counter = 0
for i in range(len(firstAs)):
    a= firstAs[i]
    try:
        href = a.get("href")
        if not href:
            continue

        fullUrl = baseUrl + href
        session = requests.Session()
        html = session.get(fullUrl, headers=headers).text

        soup2 = BeautifulSoup(html)
        aa = soup2.select_one("a[href$='.pdf']")
        if not aa:
            raise ValueError("PDF link not found")

        pdf_url = baseUrl + aa.get("href")

        if pdf_url in seen_urls:
            continue
        seen_urls.add(pdf_url)

        pdf_counter += 1
        print(f"Processing PDF #{pdf_counter}")

        # ---- load PDF ----
        loader = PyPDFLoader(pdf_url)
        docs = loader.load()

        docs = [
            d for d in docs
            if "ARRANGEMENT OF SECTIONS" not in d.page_content.upper()
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        splits = splitter.split_documents(docs)

        enriched_docs = [
            Document(
                page_content=d.page_content,
                metadata={
                    **d.metadata,
                    "doc_type": "state_act",
                    "source": "indiacode",
                    "url": pdf_url,
                    "pdf_number": pdf_counter
                }
            )
            for d in splits
        ]

        vectorstore.add_documents(enriched_docs)

    except Exception as e:
        with open(FAILED_LOG, "a") as f:
            f.write(
                f"PDF #{pdf_counter} | URL: {pdf_url if 'pdf_url' in locals() else 'UNKNOWN'} | ERROR: {str(e)}\n"
            )
        print(f" Failed PDF #{pdf_counter}")
        continue



    