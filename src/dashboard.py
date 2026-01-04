import os
from collections import Counter
from tabulate import tabulate
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# --- CONFIGURATION ---
PERSIST_DIR = "./my_local_data"
EMBED_MODEL = "nomic-embed-text"

def generate_dashboard():
    if not os.path.exists(PERSIST_DIR):
        print("Error: Data store not found. Run your ingestion script first.")
        return

    # 1. Connect to your existing store
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    
    # 2. Extract all metadata
    # We include 'metadatas' and 'documents' to get the full picture
    data = vectorstore.get(include=["metadatas"])
    metadatas = data['metadatas']
    
    if not metadatas:
        print("The data store is currently empty.")
        return

    # 3. Analyze the Data
    total_chunks = len(metadatas)
    file_stats = Counter()
    type_stats = Counter()

    for meta in metadatas:
        source = meta.get('source', 'Unknown')
        file_stats[source] += 1
        
        # Categorize by extension for your D&D vs Code split
        ext = os.path.splitext(source)[1].lower()
        if ext == ".pdf":
            type_stats["D&D / PDF Rules"] += 1
        elif ext in [".py", ".c", ".cpp", ".h"]:
            type_stats["Source Code"] += 1
        else:
            type_stats["Other Text/Docs"] += 1

    # 4. Format for Display
    print("\n" + "="*50)
    print(f"       LOCAL DATA STORE DASHBOARD (Triple F)")
    print("="*50)
    print(f"TOTAL CHUNKS IN DATABASE: {total_chunks}")
    print("-"*50)

    # Table 1: Type Breakdown
    type_table = [[k, v] for k, v in type_stats.items()]
    print("\nSTORAGE BY CATEGORY:")
    print(tabulate(type_table, headers=["Category", "Chunks"], tablefmt="presto"))

    # Table 2: Top File Sources
    # We show only the filename, not the full path, for readability
    file_table = [[os.path.basename(k), v] for k, v in file_stats.most_common(10)]
    print("\nTOP 10 SOURCES (BY CHUNK COUNT):")
    print(tabulate(file_table, headers=["Filename", "Chunks"], tablefmt="presto"))
    
    print("\n" + "="*50)

if __name__ == "__main__":
    generate_dashboard()