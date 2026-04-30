import os
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_chroma import Chroma

os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def get_relevant_context(vectorstore: Chroma, query: str, k: int = 3) -> str:
    """Given a query, retrieve the k most relevant context from the vector store and return it as a string."""
    if vectorstore is None:
        return ""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)


    if not docs:
        return ""
    # print(f"[RAG] Retrieved {len(docs)} chunks for query: {query[:50]}...")
    # for doc in docs:
    #    print(f"[RAG] Chunk: {doc.page_content[:100]}...")
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    return context