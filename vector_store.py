import os
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "/chroma_data")

def get_embedding_function():
    # Using OpenAIEmbeddings for now, but can be swapped for others if needed
    return HuggingFaceEmbeddings(groq_api_key=os.getenv("GROQ_API_KEY"))

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

def get_vector_store():
    client = get_chroma_client()
    embedding_function = get_embedding_function()
    # Ensure the collection exists or create it
    collection_name = "study_materials"
    try:
        collection = client.get_collection(name=collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    
    return Chroma(client=client, collection_name=collection_name, embedding_function=embedding_function)

def add_document_to_vector_store(file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"Added {len(chunks)} chunks from {file_path} to vector store.")

def retrieve_context(query: str, k: int = 5):
    vector_store = get_vector_store()
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])
