import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from answer import answer_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class QueryResponse(BaseModel):
    question: str
    answer: str


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        logger.info(f"Received query request: '{request.question}'")
        answer = answer_query(request.question, top_k=request.top_k)
        logger.info(f"Successfully generated answer for query")
        return QueryResponse(question=request.question, answer=answer)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    logger.info("Health check requested")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server on 0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
