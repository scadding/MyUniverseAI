import os
import time
import pathspec

# Suppress warnings
import warnings
from langchain_core._api import LangChainDeprecationWarning
# This catches the general "Chain" deprecations
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
# This catches the specific "Memory" and "Buffer" warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from git import Repo
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# LangChain v0.3 / v1.0 Imports
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import PromptTemplate

# This suppresses only the LangChain-specific 'Legacy' warnings
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

# Classic Bridge for Conversational Logic
try:
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain
except ImportError:
    from langchain_classic.memory import ConversationBufferMemory
    from langchain_classic.chains import ConversationalRetrievalChain

# --- SETTINGS ---
WATCH_DIR = "./my_data"
PERSIST_DIR = "./my_local_data"
IGNORE_FILE = ".gitignore"
EMBED_MODEL = "manutic/nomic-embed-code"
# CHAT_MODEL = "llama3"
CHAT_MODEL = "deepseek-r1:32b"


# FINAL SYSTEM PROMPT
SYSTEM_PROMPT = """
### ROLE
You are a dual-domain expert: a Professional Game Master for D&D and a Senior Software Engineer. 

### INSTRUCTIONS
1. Use the provided Context to answer the question.
2. If the answer is not in the context, simply say "I'm sorry, I don't know."
3. **CITATIONS REQUIRED**: You MUST end every response with a section titled "SOURCES USED".
   - List the file name and the git branch (if applicable) for every piece of information used.
   - Format: [Filename] - Branch: [Branch Name]

### DOMAIN RULES
- **D&D MODE**: When answering about game rules, explain the mechanics clearly. If the user asks for a 'triple f' (fire/force/form) calculation, prioritize the damage and range.
- **CODE MODE**: When answering about code, provide technical logic. Use backticks for all variable names, function names, and file names.

### CONTEXT
{context}

### CHAT HISTORY
{chat_history}

### USER QUESTION
{question}

### RESPONSE:
"""

# --- 1. INITIALIZE MODELS ---
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=CHAT_MODEL)
vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# --- 2. LOGIC FUNCTIONS ---
def get_ignore_spec():
    if os.path.exists(IGNORE_FILE):
        with open(IGNORE_FILE, "r") as f:
            return pathspec.PathSpec.from_lines('gitwildmatch', f.read().splitlines())
    return None

def get_code_splitter(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".py":
        return RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size=1000, chunk_overlap=100)
    elif ext in [".c", ".cpp", ".h", ".hpp"]:
        return RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=1000, chunk_overlap=100)
    return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def process_file(file_path, spec):
    if any(x in file_path for x in [".git", "__pycache__", "node_modules"]): return
    if spec and type(spec) != str and spec.match_file(os.path.relpath(file_path, WATCH_DIR)): return

    # Clean existing
    vectorstore._collection.delete(where={"source": os.path.abspath(file_path)})

    if os.path.exists(file_path) and not os.path.isdir(file_path):
        try:
            loader = PyPDFLoader(file_path) if file_path.endswith(".pdf") else TextLoader(file_path)
            docs = loader.load()
            
            # Get Git Branch for metadata
            branch = "N/A"
            try:
                repo = Repo(file_path, search_parent_directories=True)
                branch = repo.active_branch.name
            except: pass

            for doc in docs:
                doc.metadata["source"] = os.path.abspath(file_path)
                doc.metadata["branch"] = branch
            
            chunks = get_code_splitter(file_path).split_documents(docs)
            vectorstore.add_documents(chunks)
            print(f"Indexed: {os.path.basename(file_path)} [{branch}]")
        except: pass

# --- 3. THE CHAT ENGINE ---
def start_chat():
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    prompt = PromptTemplate(input_variables=["context", "chat_history", "question"], template=SYSTEM_PROMPT)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    while True:
        query = input("You: ")
        
        # 1. Handle Exit
        if query.lower() in ["exit", "quit"]:
            break
            
        # 2. Handle Manual Refresh Command
        if query.lower() == "/refresh":
            print("\n--- Manual Sync Triggered ---")
            # We use the same 'spec' and 'process_file' from your main script
            for root, dirs, files in os.walk(WATCH_DIR):
                for file in files:
                    full_path = os.path.join(root, file)
                    process_file(full_path, spec)
            print("--- Sync Complete. Data store is up to date. ---\n")
            continue

        if query.strip() == "/stats":
            results = vectorstore.get(include=['metadatas'])
            
            # The Safety Gate
            if results is not None and 'metadatas' in results:
                metas = results['metadatas']
                chunk_count = len(metas)
                
                # Using a generator expression for efficiency
                unique_files = len(set(m['source'] for m in metas if m and 'source' in m))
                
                print(f"\n--- Triple F Stats ---")
                print(f"Total Chunks: {chunk_count}")
                print(f"Unique Files: {unique_files}")
                print(f"----------------------\n")
            else:
                print("I'm sorry, I don't know the stats. The database returned no data.")
            continue

        # 3. Standard Chat Processing
        try:
            response = chain.invoke({"question": query})
            print(f"\nAI: {response['answer']}\n")
        except Exception as e:
            print(f"\nError: {e}")

# --- REFINED HANDLER CLASS ---
class RAGSyncHandler(FileSystemEventHandler):
    def __init__(self, spec):
        self.spec = spec
        self.last_processed = {} # {path: timestamp}
        self.debounce_seconds = 1 # 1 second is usually perfect

    def process_file(self, event):
        if event.is_directory:
            return

        full_path = os.path.abspath(event.src_path)
        current_time = time.time()

        # DEBOUNCE LOGIC
        if full_path in self.last_processed:
            if current_time - self.last_processed[full_path] < self.debounce_seconds:
                # Too soon! Skip this duplicate event.
                return 

        self.last_processed[full_path] = current_time
        
        # Now run your actual indexing logic
        print(f"Indexing: {full_path}")
        process_file(full_path, spec)
        # index_to_chroma(full_path)

    def on_modified(self, event):
        # Inappropriately overriding often misses this directory check!
        if event.is_directory:
            return
        print(f"Modification detected: {event.src_path}")
        self.process_file(event)

    def on_created(self, event):
        if event.is_directory:
            return
        print(f"New file detected: {event.src_path}")
        self.process_file(event)

    def on_deleted(self, event):
        if event.is_directory:
            return
        print(f"File deleted: {event.src_path}")
        # Clean the DB immediately when a file is gone
        remove_file_from_db(event.src_path)

def remove_file_from_db(file_path):
    try:
        # 1. Always normalize to an absolute path so the match is exact
        norm_path = os.path.abspath(file_path)
        
        # 2. Access the low-level Chroma collection
        # This bypasses the LangChain 'where' parameter constraints
        vectorstore._collection.delete(
            where={"source": norm_path}
        )
        print(f"--- Successfully purged: {os.path.basename(file_path)} ---")
    except Exception as e:
        # If the file wasn't in the DB yet, Chroma might throw an error
        # We can safely ignore it if the goal was deletion anyway
        print(f"Note: No existing records found for {os.path.basename(file_path)}")

# --- 4. RUNTIME ---
if __name__ == "__main__":
    if not os.path.exists(WATCH_DIR): 
        os.makedirs(WATCH_DIR)
        
    # This is the actual pathspec object
    spec = get_ignore_spec()

    # Start Watchdog in Background
    observer = Observer()
    handler = RAGSyncHandler(WATCH_DIR)
    observer.schedule(handler, WATCH_DIR, recursive=True)
    observer.start()
    
    # Run Initial Sync
    print("Performing initial sync...")
    #for r, d, f in os.walk(WATCH_DIR):
    #    for file in f: process_file(os.path.join(r, file), spec)

    # Launch Chat
    try:
        start_chat()
    finally:
        observer.stop()
        observer.join()

