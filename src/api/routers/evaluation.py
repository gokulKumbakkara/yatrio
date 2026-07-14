import logging
from fastapi import APIRouter, BackgroundTasks
from src.evaluation.evaluator import get_evaluator

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory evaluation status
_eval_status: dict = {"state": "idle", "results": None, "error": None}


def _run_evaluation():
    global _eval_status
    _eval_status = {"state": "running", "results": None, "error": None}
    try:
        evaluator = get_evaluator()
        results = evaluator.evaluate()
        _eval_status["state"] = "done"
        _eval_status["results"] = results
        logger.info("Evaluation complete: %s", results)
    except Exception as e:
        _eval_status["state"] = "error"
        _eval_status["error"] = str(e)
        logger.error("Evaluation failed: %s", e)


@router.post("")
async def evaluate(background_tasks: BackgroundTasks):
    """
    Trigger RAGAS evaluation pipeline in background.
    Evaluation takes several minutes — use /evaluate/status to check progress.
    """
    if _eval_status["state"] == "running":
        return {"message": "Evaluation already running"}

    background_tasks.add_task(_run_evaluation)
    return {
        "message": "Evaluation started. Use /evaluate/status to track progress.",
        "note": "This will take several minutes — 30 questions × RAG pipeline + RAGAS scoring",
    }


@router.get("/status")
async def evaluate_status():
    """Check evaluation status and results."""
    return _eval_status