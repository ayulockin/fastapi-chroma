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

@app.post("/query", response_model=QueryResponse)
def query_endpoint(body: QueryRequest):
    try:
        result = query_vectordb(body.query)
        return QueryResponse(**result)
    except Exception as e:
        # You can log e with any logger if desired
        raise HTTPException(status_code=500, detail=str(e))
