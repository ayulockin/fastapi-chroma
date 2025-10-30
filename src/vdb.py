import os
import chromadb

# Set cache directory to /tmp (only writable dir in Vercel serverless)
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/tmp/sentence_transformers"

print(f"[DEBUG] Cache directories set:")
print(f"  HF_HOME: {os.environ.get('HF_HOME')}")
print(f"  TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")
print(f"  SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME')}")

# Create the client once at import time to avoid re-init on every request
_client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ.get("CHROMA_TENANT", ""),   # optional if key is scoped
    database=os.environ.get("CHROMA_DB", ""),     # optional if key is scoped
)

print("[DEBUG] ChromaDB CloudClient initialized successfully")

def query_vectordb(query: str) -> dict:
    """
    Returns: {"context": str, "docs_id": list[str]}
    """
    print(f"[DEBUG] Querying with: {query}")
    
    collection_name = os.environ.get("CHROMA_COLLECTION", "people_ops")
    collection = _client.get_collection(name=collection_name)

    res = collection.query(query_texts=[query], n_results=3, include=["documents", "metadatas", "distances"])

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]

    # Put your own summarization/formatting here if you want.
    # For now, just join top docs with separators.
    context = "\n\n---\n\n".join(docs)
    
    print(f"[DEBUG] Query successful, returning {len(ids)} results")

    return {
        "context": context,
        "docs_id": ids,
    }


if __name__ == "__main__":
    print(query_vectordb("What are the Plan Names for UHC?"))
