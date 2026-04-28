import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_index(file_path: str) -> Chroma:
    """Load PDF, split into chunks, embed and store in memory"""

    # PDF load
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split to chunks

    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=50) # 500 words per chunk with 50 overlap to not loose context
    chunks = splitter.split_documents(documents)

    # Embed and store in memory
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, collection_name="student_upload")
    return vectorstore