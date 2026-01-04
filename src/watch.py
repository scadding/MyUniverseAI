import os
import time
import pathspec
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# --- CONFIGURATION ---
WATCH_DIR = "./my_knowledge_base"
PERSIST_DIR = "./my_local_data"
IGNORE_FILE = ".gitignore"
EMBED_MODEL = "nomic-embed-text"

# 1. Initialize Vector Store & Embeddings
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def get_ignore_spec():
    if os.path.exists(IGNORE_FILE):
        with open(IGNORE_FILE, "r") as f:
            return pathspec.PathSpec.from_lines('gitwildmatch', f.read().splitlines())
    return None

def should_ignore(file_path, spec):
    rel_path = os.path.relpath(file_path, WATCH_DIR)
    return spec and spec.match_file(rel_path)

def get_code_splitter(file_path):
    """Returns a splitter tailored to the specific programming language."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".py":
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
        )
    elif ext in [".c", ".cpp", ".h", ".hpp"]:
        return RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP, chunk_size=1000, chunk_overlap=100
        )
    # Default splitter for text/markdown/D&D rules
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def remove_file_from_db(file_path):
    try:
        # Normalize path for metadata consistency
        norm_path = os.path.abspath(file_path)
        vectorstore._collection.delete(where={"source": norm_path})
        print(f"--- Cleaned entries for: {os.path.basename(file_path)} ---")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

def process_file(file_path, spec):
    if should_ignore(file_path, spec):
        return

    # Always clear existing data for this file first
    remove_file_from_db(file_path)

    if os.path.exists(file_path):
        try:
            # Choose appropriate loader
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith((".py", ".c", ".cpp", ".h", ".hpp", ".txt", ".md")):
                loader = TextLoader(file_path)
            else:
                return

            docs = loader.load()
            # Ensure the metadata source is an absolute path for reliable deletion later
            for doc in docs:
                doc.metadata["source"] = os.path.abspath(file_path)

            # Get the specialized splitter for this file type
            splitter = get_code_splitter(file_path)
            chunks = splitter.split_documents(docs)
            
            vectorstore.add_documents(chunks)
            print(f"--- Successfully Indexed: {os.path.basename(file_path)} ---")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

# --- WATCHDOG HANDLER ---
class RAGSyncHandler(FileSystemEventHandler):
    def __init__(self, spec):
        self.spec = spec

    def on_modified(self, event):
        if not event.is_directory:
            process_file(event.src_path, self.spec)

    def on_created(self, event):
        if not event.is_directory:
            process_file(event.src_path, self.spec)

    def on_deleted(self, event):
        if not event.is_directory:
            remove_file_from_db(event.src_path)

# --- EXECUTION ---
if __name__ == "__main__":
    ignore_spec = get_ignore_spec()

    print("--- Starting Initial Full Sync ---")
    for root, dirs, files in os.walk(WATCH_DIR):
        # Skip ignored directories to save time
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_spec)]
        for file in files:
            process_file(os.path.join(root, file), ignore_spec)

    observer = Observer()
    handler = RAGSyncHandler(ignore_spec)
    observer.schedule(handler, WATCH_DIR, recursive=True)
    
    print(f"--- Watchdog Ready. Monitoring {WATCH_DIR} ---")
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()