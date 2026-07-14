import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.routers import ingestion, query
from src.api.routers import evaluation
from src.api.routers import agent



logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — initialize anything that needs to be ready before requests
    # e.g. vector DB connection, embedding model
    print("fit_yatra.ai starting up...")
    yield
    # Shutdown
    print("fit_yatra.ai shutting down...")


app = FastAPI(
    title="fit_yatra.ai",
    description="Intelligent Indian Travel & Food Companion — RAG API",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingestion.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(evaluation.router, prefix="/evaluate", tags=["Evaluation"])
app.include_router(agent.router, prefix="/agent", tags=["Agent"])



@app.get("/health")
async def health():
    return {"status": "ok", "service": "yatra-ai"}