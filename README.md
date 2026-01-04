# MyUniverseAI

How to Install
Create a Virtual Environment (Recommended):

Bash

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
Install the Libraries:

Bash

pip install -r Requirements.txt
Why these specific versions?
langchain-ollama: This is the new standard integration (moving away from the older community-only ChatOllama). It is much faster and handles "context window" limits more gracefully for your codebases.

langchain-chroma: Using the direct integration library instead of the general wrapper allows the delete(where={"source": path}) command to be significantly more reliable, which is critical for your "clean-on-change" logic.

pathspec: This is the exact library used by Git itself; it ensures that if you put *.log or build/ in your ignore file, the RAG system respects it perfectly.


# bash
export OLLAMA_NUM_PARALLEL=4
ollama restart


#########

ingest
audit





# TODO:
# 1. add sqlite to store file, mod time, and checksum
# 2. verify first before triggering RAG
# 3. logfile
# 4. UI
# 5. remove junk from myuniverse
# 4. common config file
# 5. Add threaded ingestion
# 6. prompt switchboard

