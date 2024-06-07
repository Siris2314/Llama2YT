from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from models.hosted import *
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to allow only specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    url: str
    query: str

@app.post("/process_query")
async def process_query_endpoint(request: QueryRequest):
    try:
        # Load the transcript and initialize the database
        db = data_loader(request.url)
        
        # Load the LLM model
        llm = load_tokenizer_and_llm()
        
        # Process the query
        result = process_query(request.query, llm, db)
        
        return {"answer": result["answer"], "sources": result["sources"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
