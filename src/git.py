import os
import time
import pathspec
from git import Repo
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

# --- CONFIG ---
WATCH_DIR = "./my_knowledge_base"
PERSIST_DIR = "./my_local_data"
IGNORE_FILE = ".gitignore" # Your local .gitignore file
EMBED_MODEL = "nomic-embed-text"

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

def get_ignore_spec():
    if os.path.exists(IGNORE_FILE):
        with open(IGNORE_FILE, "r") as f:
            return pathspec.PathSpec.from_lines('gitwildmatch', f.read().splitlines())
    return None

def should_ignore(file_path, spec):
    rel_path = os.path.relpath(file_path, WATCH_DIR)
    # Automatically ignore the .git folder to avoid indexing internal git data
    if ".git" in rel_path.split(os.sep): return True
    return spec and spec.match_file(rel_path)

def get_code_splitter(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".py":
        return RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1000, chunk_overlap=100)
    elif ext in [".c", ".cpp", ".h", ".hpp"]:
        return RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=1000, chunk_overlap=100)
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def remove_file_from_db(file_path):
    try:
        norm_path = os.path.abspath(file_path)
        vectorstore._collection.delete(where={"source": norm_path})
        print(f"--- Cleaned: {os.path.basename(file_path)} ---")
    except Exception as e:
        print(f"Error cleaning {file_path}: {e}")

def process_file(file_path, spec):
    if should_ignore(file_path, spec) or os.path.isdir(file_path):
        return

    remove_file_from_db(file_path)

    if os.path.exists(file_path):
        try:
            if file_path.endswith(".pdf"): loader = PyPDFLoader(file_path)
            elif file_path.endswith((".py", ".c", ".cpp", ".h", ".hpp", ".txt", ".md")): loader = TextLoader(file_path)
            else: return

            docs = loader.load()
            for doc in docs: doc.metadata["source"] = os.path.abspath(file_path)
            
            splitter = get_code_splitter(file_path)
            chunks = splitter.split_documents(docs)
            vectorstore.add_documents(chunks)
            print(f"--- Indexed: {os.path.basename(file_path)} ---")
        except Exception as e:
            print(f"Failed {file_path}: {e}")

"""
update for checking
def process_file(file_path, spec):
    # ... existing ignore checks ...
    
    # Try to get the current git branch if the file is in a repo
    current_branch = "N/A"
    try:
        repo = Repo(file_path, search_parent_directories=True)
        current_branch = repo.active_branch.name
    except:
        pass # Not a git repo

    # ... in the ingestion part ...
    for doc in docs:
        doc.metadata["source"] = os.path.abspath(file_path)
        doc.metadata["branch"] = current_branch  # <--- New metadata tag
"""


def clone_github_repo(repo_url):
    """Clones a GitHub repo into the watch directory."""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    target_path = os.path.join(WATCH_DIR, repo_name)
    
    if not os.path.exists(target_path):
        print(f"--- Cloning {repo_url} ---")
        Repo.clone_from(repo_url, target_path)
    else:
        print(f"--- Repo {repo_name} already exists. Pulling latest... ---")
        Repo(target_path).remotes.origin.pull()

def ingest_github_repo(repo_url, branch="main"):
    """
    Clones a specific branch or switches an existing repo to a new branch.
    """
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    target_path = os.path.join(WATCH_DIR, repo_name)
    
    if not os.path.exists(target_path):
        print(f"--- Cloning {repo_url} (Branch: {branch}) ---")
        # Use single_branch=True to keep your local storage light
        Repo.clone_from(repo_url, target_path, branch=branch, single_branch=True)
    else:
        repo = Repo(target_path)
        print(f"--- Switching {repo_name} to branch: {branch} ---")
        
        # Fetch and checkout the branch
        repo.remotes.origin.fetch()
        try:
            repo.git.checkout(branch)
            repo.remotes.origin.pull()
        except Exception as e:
            print(f"Branch switch failed: {e}")
            return

    # Note: The Watchdog will automatically notice the files changing 
    # during the checkout/pull and trigger the re-indexing for you.

# --- WATCHDOG HANDLER ---
class RAGSyncHandler(FileSystemEventHandler):
    def __init__(self, spec): self.spec = spec
    def on_modified(self, event): 
        if not event.is_directory: process_file(event.src_path, self.spec)
    def on_created(self, event):
        if not event.is_directory: process_file(event.src_path, self.spec)
    def on_deleted(self, event):
        if not event.is_directory: remove_file_from_db(event.src_path)

if __name__ == "__main__":
    if not os.path.exists(WATCH_DIR): os.makedirs(WATCH_DIR)
    ignore_spec = get_ignore_spec()

    # Example: To ingest a repo, uncomment the line below or call it as needed
    # clone_github_repo("https://github.com/langchain-ai/langchain")

    print("--- Running Initial Sync ---")
    for root, dirs, files in os.walk(WATCH_DIR):
        for file in files:
            process_file(os.path.join(root, file), ignore_spec)

    observer = Observer()
    handler = RAGSyncHandler(ignore_spec)
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.start()
    
    print(f"--- Watchdog Active on {WATCH_DIR} ---")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()