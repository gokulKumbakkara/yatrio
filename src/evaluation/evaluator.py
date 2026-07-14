import time
import logging
from groq import RateLimitError, APIConnectionError, InternalServerError
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.settings import settings
from src.ingestion.loader import get_loader
from src.api.routers.query import get_pipeline, QueryRequest

logger = logging.getLogger(__name__)

QUESTIONS_PER_CATEGORY = 1
CATEGORIES = ["travel", "nutrition", "food"]

# Retry configuration for Groq API calls
_GROQ_MAX_RETRIES = 3
_GROQ_INITIAL_BACKOFF = 1.0  # seconds
_INTER_REQUEST_DELAY = 1.0   # seconds between successful API calls


class RAGEvaluator:
    """
    Evaluates Yatra.ai RAG pipeline using RAGAS metrics:
      - Faithfulness     : is answer grounded in retrieved context?
      - Answer Relevancy : is answer relevant to the question?
      - Context Precision: are retrieved chunks relevant to the question?

    Uses Groq LLM + HuggingFace embeddings — no OpenAI needed.
    """

    def __init__(self):
        logger.info("Initialising RAGAS evaluator...")

        # Wrap Groq as RAGAS LLM
        self.ragas_llm = LangchainLLMWrapper(
            ChatGroq(
                api_key=settings.groq_api_key,
                model_name=settings.groq_model,
            )
        )

        # Wrap HuggingFace embeddings as RAGAS embeddings
        self.ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(
                model_name=settings.embedding_model_id,
            )
        )

        # Configure metrics to use Groq + HuggingFace
        self.metrics = [faithfulness, answer_relevancy]
        for metric in self.metrics:
            metric.llm = self.ragas_llm
            if hasattr(metric, "embeddings"):
                metric.embeddings = self.ragas_embeddings

        logger.info("RAGAS evaluator ready")

    # ── Step 1: Generate Synthetic Q&A Dataset ────────────────────────────────

    def _generate_questions_for_category(
        self, category: str, n: int
    ) -> list[dict]:
        """
        Fetch sample chunks from ChromaDB for a category.
        Use Groq to generate Q&A pairs from those chunks.
        """
        loader = get_loader()

        try:
            results = loader.db._collection.get(
                where={"category": category},
                include=["documents"],
                limit=n * 3,
            )
        except Exception as e:
            logger.error("Failed to fetch chunks for category '%s': %s", category, e)
            return []

        chunks = results["documents"]
        if not chunks:
            logger.warning("No chunks found for category: %s", category)
            return []

        qa_pairs = []
        pipeline = get_pipeline()

        for chunk in chunks[:n]:
            prompt = f"""Generate a question and answer pair from this text.
The question should be answerable from the text.
Return ONLY in this format:
QUESTION: <question>
ANSWER: <answer>

Text: {chunk}"""

            backoff = _GROQ_INITIAL_BACKOFF
            for attempt in range(_GROQ_MAX_RETRIES):
                try:
                    response = pipeline.groq.chat.completions.create(
                        model=settings.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=200,
                    )

                    content = response.choices[0].message.content.strip()
                    lines = content.split("\n")

                    question = ""
                    answer = ""
                    for line in lines:
                        if line.startswith("QUESTION:"):
                            question = line.replace("QUESTION:", "").strip()
                        elif line.startswith("ANSWER:"):
                            answer = line.replace("ANSWER:", "").strip()

                    if question and answer:
                        qa_pairs.append({
                            "question": question,
                            "ground_truth": answer,
                            "category": category,
                        })

                    # Successful call — apply inter-request delay and move on
                    time.sleep(_INTER_REQUEST_DELAY)
                    break

                except RateLimitError:
                    if attempt == _GROQ_MAX_RETRIES - 1:
                        logger.warning(
                            "Rate limit hit for category '%s' chunk — skipping after %d attempts",
                            category, _GROQ_MAX_RETRIES,
                        )
                        break
                    wait = backoff * (2 ** attempt)
                    logger.warning(
                        "Rate limit hit — waiting %.1fs (attempt %d/%d)",
                        wait, attempt + 1, _GROQ_MAX_RETRIES,
                    )
                    time.sleep(wait)

                except (APIConnectionError, InternalServerError) as e:
                    if attempt == _GROQ_MAX_RETRIES - 1:
                        logger.warning(
                            "API error for category '%s' chunk after %d attempts: %s",
                            category, _GROQ_MAX_RETRIES, e,
                        )
                        break
                    wait = backoff * (2 ** attempt)
                    logger.warning(
                        "API error — waiting %.1fs (attempt %d/%d): %s",
                        wait, attempt + 1, _GROQ_MAX_RETRIES, e,
                    )
                    time.sleep(wait)

                except Exception as e:
                    logger.warning("Failed to generate Q&A for chunk: %s", e)
                    break

        logger.info(
            "Generated %d Q&A pairs for category: %s",
            len(qa_pairs), category,
        )
        return qa_pairs

    def generate_eval_dataset(self) -> list[dict]:
        """Generate synthetic Q&A pairs across all categories."""
        all_qa_pairs = []
        for category in CATEGORIES:
            pairs = self._generate_questions_for_category(
                category, QUESTIONS_PER_CATEGORY
            )
            all_qa_pairs.extend(pairs)

        logger.info("Total Q&A pairs generated: %d", len(all_qa_pairs))
        return all_qa_pairs

    # ── Step 2: Run RAG Pipeline on Each Question ─────────────────────────────

    def _run_pipeline_on_dataset(self, qa_pairs: list[dict]) -> list[dict]:
        """
        Run RAG pipeline on each question.
        Collect: question, answer, contexts, ground_truth.
        """
        pipeline = get_pipeline()
        loader = get_loader()
        eval_records = []

        for i, qa in enumerate(qa_pairs):
            logger.info(
                "Running pipeline on question %d/%d: %s",
                i + 1, len(qa_pairs), qa["question"][:50],
            )
            try:
                request = QueryRequest(
                    question=qa["question"],
                    category=qa["category"],
                    top_k=settings.final_top_k,
                )
                response = pipeline.ask(request)

                candidates = pipeline._search(
                    [qa["question"]],
                    top_k=settings.rerank_candidates,
                    category=[qa["category"]],
                    chunk_type=None,
                )
                top_chunks = pipeline._rerank(
                    qa["question"], candidates, top_k=settings.final_top_k
                )
                contexts = [chunk.text for chunk in top_chunks]

                eval_records.append({
                    "question": qa["question"],
                    "answer": response.answer,
                    "contexts": contexts,
                    "ground_truth": qa["ground_truth"],
                })

            except Exception as e:
                logger.warning("Pipeline failed for question: %s — %s", qa["question"], e)
                continue

        logger.info("Pipeline run complete — %d records", len(eval_records))
        return eval_records

    # ── Step 3: Run RAGAS Metrics ─────────────────────────────────────────────

    def evaluate(self) -> dict:
        """
        Full evaluation flow:
          1. Generate synthetic Q&A dataset
          2. Run RAG pipeline on each question
          3. Score with RAGAS metrics
          4. Return results
        """
        logger.info("Starting RAGAS evaluation...")

        # Step 1 — Generate dataset
        try:
            qa_pairs = self.generate_eval_dataset()
        except Exception as e:
            logger.error("Failed to generate eval dataset: %s", e)
            return {"error": f"Dataset generation failed: {e}"}

        if not qa_pairs:
            return {"error": "No Q&A pairs generated"}

        # Step 2 — Run pipeline
        try:
            eval_records = self._run_pipeline_on_dataset(qa_pairs)
        except Exception as e:
            logger.error("Failed to run pipeline on dataset: %s", e)
            return {"error": f"Pipeline execution failed: {e}"}

        time.sleep(2)

        if not eval_records:
            return {"error": "Pipeline failed on all questions"}

        # Step 3 — Convert to RAGAS Dataset format
        try:
            ragas_dataset = Dataset.from_list(eval_records)
        except Exception as e:
            logger.error("Failed to build RAGAS dataset: %s", e)
            return {"error": f"RAGAS dataset construction failed: {e}"}

        # Step 4 — Run RAGAS metrics
        logger.info("Running RAGAS metrics on %d records...", len(eval_records))
        try:
            results = evaluate(
                dataset=ragas_dataset,
                metrics=self.metrics,
            )
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            return {"error": f"RAGAS scoring failed: {e}"}

        # Step 5 — Format results
        try:
            scores = results.to_pandas().mean().to_dict()
        except Exception as e:
            logger.error("Failed to parse RAGAS results: %s", e)
            return {"error": f"Failed to parse RAGAS results: {e}"}

        output = {
            "total_questions": len(eval_records),
            "metrics": {
                "faithfulness": round(scores.get("faithfulness", 0), 4),
                "answer_relevancy": round(scores.get("answer_relevancy", 0), 4)
            },
            "interpretation": self._interpret_scores(scores),
        }

        logger.info("RAGAS evaluation complete: %s", output["metrics"])
        return output

    def _interpret_scores(self, scores: dict) -> dict:
        """
        Interpret RAGAS scores into actionable insights.
        Score ranges: 0 (worst) → 1 (best)
        """
        def interpret(score: float, metric: str) -> str:
            if score >= 0.8:
                return f"{metric} is excellent"
            elif score >= 0.6:
                return f"{metric} is good but can be improved"
            elif score >= 0.4:
                return f"{metric} needs improvement — check chunking/retrieval"
            else:
                return f"{metric} is poor — significant issues in pipeline"

        return {
            "faithfulness": interpret(scores.get("faithfulness", 0), "Faithfulness"),
            "answer_relevancy": interpret(scores.get("answer_relevancy", 0), "Answer relevancy"),
            "context_precision": interpret(scores.get("context_precision", 0), "Context precision"),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_evaluator: RAGEvaluator | None = None


def get_evaluator() -> RAGEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGEvaluator()
    return _evaluator
