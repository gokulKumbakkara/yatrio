import json
import logging
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException

from src.ingestion.parser import DocumentParser
from src.ingestion.cleaner import clean_documents 
from src.ingestion.chunker import chunk_documents 
from src.ingestion.embedder import embed_chunks
from src.ingestion.loader import load_to_vectordb, get_loader

router = APIRouter()
logger = logging.getLogger(__name__)

_status: dict = {"state": "idle", "docs_processed": 0, "error": None}


def _run_pipeline() -> None:
    global _status
    _status = {"state": "running", "docs_processed": 0, "error": None}
    logger.info("Ingestion pipeline started")
    try:
        # Step 1: Parse
        _status["state"] = "parsing"
        parser = DocumentParser()
        docs = parser.parse_all()

        # Step 2: Clean          ← add from here
        _status["state"] = "cleaning"
        cleaned_docs = clean_documents(docs)
        _status["docs_processed"] = len(cleaned_docs)

        # Step 3: Chunk (coming next)
        _status["state"] = "chunking"
        chunks = chunk_documents(cleaned_docs)
        _status["chunks_created"] = len(chunks)
        # Step 4: Embed
        # Step 4: Embed
        _status["state"] = "embedding"
        embedded_chunks = embed_chunks(chunks)
        _status["chunks_embedded"] = len(embedded_chunks)
        # Step 5: Load
        _status["state"] = "loading"
        load_to_vectordb(embedded_chunks)
        _status["docs_processed"] = len(docs)

        _status["state"] = "done"
        logger.info("Ingestion pipeline done: %d doc(s) processed", len(cleaned_docs))
    except Exception as e:
        _status["state"] = "error"
        _status["error"] = str(e)
        logger.error("Ingestion pipeline failed: %s", e)


@router.post("")
async def ingest(background_tasks: BackgroundTasks):
    if _status["state"] == "running":
        return {"message": "Ingestion already running"}
    background_tasks.add_task(_run_pipeline)
    return {"message": "Ingestion started"}


@router.get("/ingest/status")
async def ingest_status():
    return _status


@router.get("/vectordb/stats")
async def vectordb_stats():
    try:
        return get_loader().get_stats()
    except Exception as e:
        logger.error("Failed to get vectordb stats: %s", e)
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics.")


@router.get("/documents")
async def documents():
    try:
        parsed_dir = Path("data/parsed")
        if not parsed_dir.exists():
            return []

        docs = []
        for meta_file in parsed_dir.rglob("*_meta.json"):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                docs.append({
                    "file_name": meta["file_name"],
                    "category": meta["category"],
                    "source_path": meta["source_path"],
                    "metadata": meta["metadata"],
                })
            except Exception as e:
                logger.warning("Failed to read metadata file %s: %s", meta_file, e)
                continue
        return docs
    except Exception as e:
        logger.error("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail="Failed to list documents.")
