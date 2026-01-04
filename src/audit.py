import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# --- SETTINGS (Must match your ingest.py) ---
CHROMA_PATH = "./my_local_data"
CODE_MODEL = "manutic/nomic-embed-code"

def audit_triple_f():
    print(f"--- PROBING TRIPLE F: {CODE_MODEL} ---")
    
    # Initialize the librarian
    embedder = OllamaEmbeddings(model=CODE_MODEL)
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedder,
        collection_metadata={"hnsw:space": "cosine"} # Forces Cosine Similarity
    )
    # 1. THE TEST QUERY
    # We are looking for the 'Bridge' logic specifically
    query = "Where is the Python ctypes interface defined for calling C binaries?"
    
    print(f"Targeting Logic: '{query}'\n")

    # 2. PERFORM RETRIEVAL
    # We pull the top 3 most relevant chunks
    results = db.similarity_search_with_relevance_scores(query, k=3)

    if not results:
        print("FAIL: No results found. The index might be empty or corrupted.")
        return

    for i, (doc, score) in enumerate(results):
        print(f"--- RESULT #{i+1} (Confidence: {score:.4f}) ---")
        print(f"SOURCE: {doc.metadata.get('source', 'Unknown')}")
        print(f"TYPE: {doc.metadata.get('type', 'Unknown')}")
        print("-" * 30)
        # Show the first 500 characters to verify the 'Header Injection' worked
        print(doc.page_content[:500] + "...")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    audit_triple_f()