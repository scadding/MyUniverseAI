import os
import gc
import torch
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# --- SETTINGS ---
CHROMA_PATH = "./my_local_data"
SOURCE_DIR = "./my_data/MyUniverse_V2"
CODE_MODEL = "manutic/nomic-embed-code"
TEXT_MODEL = "nomic-embed-text"

def ingest_universe():
    # 1. Start Fresh
    if os.path.exists(CHROMA_PATH):
        print(f"Purging old database at {CHROMA_PATH}...")
        import shutil
        shutil.rmtree(CHROMA_PATH)

    # 2. Gather Files
    all_files = [Path(os.path.join(dp, f)) for dp, dn, filenames in os.walk(SOURCE_DIR) for f in filenames]
    code_files = [f for f in all_files if f.suffix in ['.py', '.c', '.cpp', '.h']]
    text_files = [f for f in all_files if f.suffix in ['.txt', '.md', '.pdf']]

    print(f"Found {len(code_files)} code files and {len(text_files)} text files.")

    # --- PASS 1: CODE (The Heavy Lift) ---
    if code_files:
        print(f"\n>>> Starting CODE Pass with {CODE_MODEL}...")
        code_embedder = OllamaEmbeddings(model=CODE_MODEL)
        db = Chroma(
            persist_directory=CHROMA_PATH, 
            embedding_function=code_embedder,
            collection_metadata={"hnsw:space": "cosine"} # Forces Cosine Similarity
        )        
        for i, file_path in enumerate(code_files):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Context-Aware Splitting
                ext = file_path.suffix[1:]
                lang_map = {'py': Language.PYTHON, 'c': Language.CPP, 'cpp': Language.CPP, 'h': Language.CPP}
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=lang_map.get(ext, Language.PYTHON),
                    chunk_size=1000, chunk_overlap=100
                )
                
                raw_chunks = splitter.split_text(content)
                # Inject Header so R1 always knows the file
                docs = [f"### FILE: {file_path.name}\n{chunk}" for chunk in raw_chunks]
                
                db.add_texts(docs, metadatas=[{"type": "code", "source": str(file_path)}] * len(docs))
                print(f"[{i+1}/{len(code_files)}] Indexed Code: {file_path.name}")
                
            except Exception as e:
                print(f"Skipping {file_path.name}: {e}")

        # Clear VRAM for the next model
        del code_embedder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # --- PASS 2: TEXT ---
    if text_files:
        print(f"\n>>> Starting TEXT Pass with {TEXT_MODEL}...")
        text_embedder = OllamaEmbeddings(model=TEXT_MODEL)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=text_embedder)
        
        for i, file_path in enumerate(text_files):
            # (Similar logic to above but with standard splitter)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            # ... [process and add to db] ...
            print(f"[{i+1}/{len(text_files)}] Indexed Text: {file_path.name}")

    print("\n--- Triple F Reconstruction Complete ---")

if __name__ == "__main__":
    ingest_universe()

