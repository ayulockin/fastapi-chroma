import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .vdb import query_vectordb

app = FastAPI(title="Chroma FastAPI on Vercel")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    context: str
    docs_id: list[str]

@app.get("/", tags=["health"])
def root():
    return {"ok": True}

@app.get("/debug", tags=["debug"])
def debug_info():
    """Check environment and file system status"""
    import tempfile
    
    debug_data = {
        "env_vars": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE"),
            "SENTENCE_TRANSFORMERS_HOME": os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
            "HOME": os.environ.get("HOME"),
            "TMPDIR": os.environ.get("TMPDIR"),
        },
        "temp_dir": tempfile.gettempdir(),
        "writable_test": None
    }
    
    # Test if /tmp is writable
    try:
        test_file = "/tmp/test_write.txt"
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        debug_data["writable_test"] = "/tmp is writable ✓"
    except Exception as e:
        debug_data["writable_test"] = f"/tmp write failed: {str(e)}"
    
    return debug_data

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    try:
        result = query_vectordb(body.query)
        return QueryResponse(**result)
    except Exception as e:
        # You can log e with any logger if desired
        raise HTTPException(status_code=500, detail=str(e))
