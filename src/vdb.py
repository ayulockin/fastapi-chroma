import os
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Set cache directory to /tmp (only writable location on Vercel)
os.environ["CHROMA_CACHE_DIR"] = "/tmp/chroma"

_client = chromadb.CloudClient(
    api_key=os.environ["CHROMA_API_KEY"],
    tenant=os.environ.get("CHROMA_TENANT", ""),   # optional if key is scoped
    database=os.environ.get("CHROMA_DB", ""),     # optional if key is scoped
    settings=Settings(
        is_persistent=False,  # Disable persistence in serverless environment
        anonymized_telemetry=False
    )
)


def query_vectordb(query: str) -> dict:
    """
    Returns: {"context": str, "docs_id": list[str]}
    """    
    collection_name = os.environ.get("CHROMA_COLLECTION", "people_ops_openai")
    collection = _client.get_collection(
        name=collection_name,
        embedding_function=OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
    )

    res = collection.query(query_texts=[query], n_results=3, include=["documents", "metadatas", "distances"])

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]

    # Put your own summarization/formatting here if you want.
    # For now, just join top docs with separators.
    context = "\n\n---\n\n".join(docs)

    return {
        "context": context,
        "docs_id": ids,
    }


if __name__ == "__main__":
    print(query_vectordb("What are the Plan Names for UHC?"))
