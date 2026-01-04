import chromadb
import os

# Ensure this matches your actual path
CHROMA_PATH = "./my_local_data" 

if not os.path.exists(CHROMA_PATH):
    print(f"ERROR: Path {CHROMA_PATH} does not exist.")
    exit()

client = chromadb.PersistentClient(path=CHROMA_PATH)

try:
    # List all collections to see if 'MyUniverse_V2' is actually there
    collections = client.list_collections()
    print(f"Collections found: {[c.name for c in collections]}")

    collection = client.get_collection(name="my_data")
    
    # Get EVERYTHING but only the metadata to save memory
    all_data = collection.get(include=['metadatas'])
    
    if not all_data['metadatas']:
        print("THE DATABASE IS EMPTY. Ingest failed to commit.")
    else:
        py_files = [m.get('source') for m in all_data['metadatas'] if '.py' in str(m.get('source'))]
        c_files = [m.get('source') for m in all_data['metadatas'] if '.c' in str(m.get('source'))]
        
        print(f"--- DATABASE CONTENT ---")
        print(f"Total Chunks: {len(all_data['metadatas'])}")
        print(f"Python Chunks: {len(py_files)}")
        print(f"C Chunks: {len(c_files)}")
        
        if py_files:
            print(f"Sample Python Source: {py_files[0]}")

except Exception as e:
    print(f"An error occurred: {e}")