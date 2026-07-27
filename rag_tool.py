from vector_store import add_document_to_vector_store, retrieve_context

def upload_syllabus_or_notes(file_path: str):
    """
    Uploads a PDF file (syllabus or notes) to the vector store for RAG.
    """
    try:
        add_document_to_vector_store(file_path)
        return {"status": "success", "message": f"Successfully uploaded and processed {file_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def get_rag_context(query: str):
    """
    Retrieves relevant context from the vector store based on a query.
    """
    try:
        context = retrieve_context(query)
        return {"status": "success", "context": context}
    except Exception as e:
        return {"status": "error", "message": str(e)}
