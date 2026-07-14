import re
import time
import logging
import tiktoken
from groq import Groq, RateLimitError, APIConnectionError, InternalServerError
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
from src.settings import settings
from src.ingestion.loader import get_loader, RetrievedChunk

router = APIRouter()
logger = logging.getLogger(__name__)

HIGH_STAKES_CATEGORIES = {"nutrition"}
LOW_STAKES_CATEGORIES = {"travel", "food"}

SYSTEM_PROMPT = """You are Yatra.ai, an intelligent Indian travel and food companion.
You help users with:
- Indian travel destinations, tips, and itineraries
- Indian food, recipes, and nutritional information
- Healthy eating advice tailored to Indian cuisine

Answer based on the context provided. If the context does not contain enough
information, say so honestly. Always be specific and practical in your answers.
Keep responses concise and helpful."""

# Retry configuration for Groq API calls
_GROQ_MAX_RETRIES = 3
_GROQ_INITIAL_BACKOFF = 1.0  # seconds


# ── Request / Response ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    category: str | None = None
    top_k: int = settings.final_top_k


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int
    cached: bool = False
    hallucination_check: str = "not_run"


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full RAG pipeline with:
      - Multi-query retrieval
      - Hybrid search (BM25 + vector + RRF)
      - Cross-encoder reranking
      - Context window budgeting (tiktoken)
      - Three-layer hallucination prevention
      - Streaming via SSE
    """

    def __init__(self):
        logger.info("Initialising RAG pipeline...")

        self.groq = Groq(api_key=settings.groq_api_key)

        logger.info("Loading reranker: %s", settings.reranking_model_id)
        self.reranker = CrossEncoder(settings.reranking_model_id)

        # Production token counter
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Hallucination thresholds
        self.RETRIEVAL_CONFIDENCE_THRESHOLD = -5.0
        self.HEURISTIC_OVERLAP_THRESHOLD = 0.15

        logger.info("RAG pipeline ready")

    # ── Groq API with retry ───────────────────────────────────────────────────

    def _call_groq_with_retry(self, **kwargs):
        """Call Groq API with exponential backoff on rate limit / server errors."""
        backoff = _GROQ_INITIAL_BACKOFF
        for attempt in range(_GROQ_MAX_RETRIES):
            try:
                return self.groq.chat.completions.create(**kwargs)
            except RateLimitError:
                if attempt == _GROQ_MAX_RETRIES - 1:
                    raise
                wait = backoff * (2 ** attempt)
                logger.warning(
                    "Groq rate limit hit — waiting %.1fs (attempt %d/%d)",
                    wait, attempt + 1, _GROQ_MAX_RETRIES,
                )
                time.sleep(wait)
            except (APIConnectionError, InternalServerError) as e:
                if attempt == _GROQ_MAX_RETRIES - 1:
                    raise
                wait = backoff * (2 ** attempt)
                logger.warning(
                    "Groq API error — waiting %.1fs (attempt %d/%d): %s",
                    wait, attempt + 1, _GROQ_MAX_RETRIES, e,
                )
                time.sleep(wait)

    # ── Token counting ────────────────────────────────────────────────────────

    def _count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    # ── Step 1: Multi-query ───────────────────────────────────────────────────

    def _generate_query_variations(self, question: str, n: int = 3) -> list[str]:
        prompt = f"""Generate {n} different search queries for the following question.
Each query should approach the question from a different angle.
Return only the queries, one per line, no numbering or explanations.

Question: {question}"""

        try:
            response = self._call_groq_with_retry(
                model=settings.groq_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=200,
            )
            variations = response.choices[0].message.content.strip().split("\n")
            variations = [v.strip() for v in variations if v.strip()]
            all_queries = [question] + variations[:n]
            logger.info("Generated %d query variations", len(all_queries))
            return all_queries
        except Exception as e:
            logger.warning(
                "Multi-query generation failed — falling back to original question: %s", e
            )
            return [question]

    # ── Step 2: Hybrid Search ─────────────────────────────────────────────────

    def _search(
        self,
        queries: list[str],
        top_k: int,
        category: list[str] | None,
        chunk_type: str | None,
    ) -> list[RetrievedChunk]:
        loader = get_loader()

        filters = []
        if category and len(category) == 1:
            filters.append({"category": category[0]})
        elif category and len(category) > 1:
            filters.append({"$or": [{"category": c} for c in category]})
        if chunk_type:
            filters.append({"chunk_type": chunk_type})

        where = None
        if len(filters) == 1:
            where = filters[0]
        elif len(filters) > 1:
            where = {"$and": filters}

        seen_texts: set[str] = set()
        all_chunks: list[RetrievedChunk] = []

        for query in queries:
            results = loader.hybrid_search(query, top_k, where)
            for chunk in results:
                if chunk.text not in seen_texts:
                    seen_texts.add(chunk.text)
                    all_chunks.append(chunk)

        logger.info(
            "Retrieved %d unique chunks across %d queries",
            len(all_chunks), len(queries),
        )
        return all_chunks

    # ── Step 3: Rerank ────────────────────────────────────────────────────────

    def _rerank(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [[question, chunk.text] for chunk in chunks]
        scores = self.reranker.predict(pairs)

        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        reranked = sorted(chunks, key=lambda x: x.score, reverse=True)
        logger.info("Reranked %d chunks → keeping top %d", len(chunks), top_k)
        return reranked[:top_k]

    # ── Step 4: Build Prompt ──────────────────────────────────────────────────

    def _build_prompt(self, question: str, chunks: list[RetrievedChunk]) -> str:
        TOTAL_BUDGET = 8192
        SYSTEM_TOKENS = self._count_tokens(SYSTEM_PROMPT)
        QUESTION_TOKENS = self._count_tokens(question)
        ANSWER_BUFFER = 1024
        SEPARATOR_TOKENS = 50

        available = TOTAL_BUDGET - SYSTEM_TOKENS - QUESTION_TOKENS - ANSWER_BUFFER - SEPARATOR_TOKENS

        selected_chunks = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = self._count_tokens(chunk.text)
            if used_tokens + chunk_tokens <= available:
                selected_chunks.append(chunk)
                used_tokens += chunk_tokens
            else:
                logger.warning(
                    "Budget exhausted — dropping chunk %d onwards. Used: %d/%d tokens",
                    len(selected_chunks), used_tokens, available,
                )
                break

        logger.info(
            "Context budget — chunks: %d, tokens used: %d/%d",
            len(selected_chunks), used_tokens, available,
        )

        context = "\n\n---\n\n".join(chunk.text for chunk in selected_chunks)
        return (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

    # ── Hallucination Prevention ──────────────────────────────────────────────

    def _check_retrieval_confidence(self, chunks: list[RetrievedChunk]) -> bool:
        if not chunks:
            return False
        best_score = max(chunk.score for chunk in chunks)
        logger.info("Best retrieval score: %.6f, threshold: %.6f", best_score, self.RETRIEVAL_CONFIDENCE_THRESHOLD)
        confident = best_score >= self.RETRIEVAL_CONFIDENCE_THRESHOLD
        if not confident:
            logger.warning(
                "Low retrieval confidence — best score: %.4f < threshold: %.4f",
                best_score, self.RETRIEVAL_CONFIDENCE_THRESHOLD,
            )
        return confident

    def _heuristic_hallucination_check(
        self, answer: str, chunks: list[RetrievedChunk]
    ) -> bool:
        context = " ".join(chunk.text.lower() for chunk in chunks)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "in", "on",
            "at", "to", "for", "of", "and", "or", "but", "it", "this",
            "that", "with", "as", "by", "from", "be", "has", "have",
            "had", "not", "they", "we", "you", "i", "he", "she"
        }
        answer_words = set(
            word.lower()
            for word in re.findall(r'\b\w+\b', answer)
            if word.lower() not in stopwords and len(word) > 3
        )
        if not answer_words:
            return True
        overlap = sum(1 for word in answer_words if word in context)
        overlap_ratio = overlap / len(answer_words)
        grounded = overlap_ratio >= self.HEURISTIC_OVERLAP_THRESHOLD
        logger.info(
            "Heuristic check — overlap: %.2f (%d/%d words) — grounded: %s",
            overlap_ratio, overlap, len(answer_words), grounded,
        )
        return grounded

    def _llm_judge_hallucination(
        self, question: str, answer: str, chunks: list[RetrievedChunk]
    ) -> bool:
        context = "\n\n".join(chunk.text for chunk in chunks)
        judge_prompt = f"""You are a fact-checking assistant. Verify if the answer is supported by the context.

Context:
{context}

Question: {question}

Answer: {answer}

Is the answer fully supported by the context?
Reply with ONLY 'YES' if grounded, or 'NO' if it contains information not in the context."""

        try:
            response = self._call_groq_with_retry(
                model=settings.groq_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                max_tokens=10,
            )
            verdict = response.choices[0].message.content.strip().upper()
            grounded = verdict.startswith("YES")
            logger.info("LLM judge verdict: %s", verdict)
            return grounded
        except Exception as e:
            logger.warning("LLM judge failed — treating as not grounded: %s", e)
            return False

    def _check_hallucination(
        self, question: str, answer: str, chunks: list[RetrievedChunk]
    ) -> tuple[bool, str]:
        if not self._heuristic_hallucination_check(answer, chunks):
            return False, "heuristic_fail"
        if not self._llm_judge_hallucination(question, answer, chunks):
            return False, "llm_judge_fail"
        return True, "grounded"

    # ── Detect helpers ────────────────────────────────────────────────────────

    def _detect_categories(self, question: str) -> list[str] | None:
        question_lower = question.lower()
        categories = []
        if any(k in question_lower for k in ["travel", "visit", "trip", "destination", "tourism", "place", "hotel", "city"]):
            categories.append("travel")
        if any(k in question_lower for k in ["healthy", "nutrition", "diet", "calories", "protein", "vitamin", "nutrient"]):
            categories.append("nutrition")
        if any(k in question_lower for k in ["recipe", "cook", "dish", "food", "eat", "meal", "ingredient", "prepare"]):
            categories.append("food")
        return categories if categories else None

    def _detect_chunk_type(self, question: str) -> str | None:
        table_keywords = ["how much", "how many", "quantity", "amount", "content",
                          "value", "percentage", "grams", "calories", "nutrients", "table", "list"]
        if any(k in question.lower() for k in table_keywords):
            return "table"
        return None

    # ── Main: Full Pipeline ───────────────────────────────────────────────────

    def ask(self, request: QueryRequest) -> QueryResponse:
        """Full pipeline with multi-query, hybrid search, reranking, hallucination check."""
        logger.info("Query: %s", request.question)
        try:
            category = [request.category] if request.category else self._detect_categories(request.question)
            chunk_type = self._detect_chunk_type(request.question)

            # Step 1 — Multi-query
            queries = self._generate_query_variations(request.question)

            # Step 2 — Hybrid search
            candidates = self._search(
                queries,
                top_k=settings.rerank_candidates,
                category=category,
                chunk_type=chunk_type,
            )

            if not candidates:
                return QueryResponse(
                    answer="I could not find relevant information to answer your question.",
                    sources=[], chunks_used=0,
                )

            # Step 3 — Rerank
            top_chunks = self._rerank(request.question, candidates, top_k=request.top_k)

            # Layer 1 — Retrieval confidence check
            if not self._check_retrieval_confidence(top_chunks):
                return QueryResponse(
                    answer="I don't have enough information in my knowledge base to answer this question.",
                    sources=[], chunks_used=0, hallucination_check="low_confidence",
                )

            # Step 4 — Build prompt
            prompt = self._build_prompt(request.question, top_chunks)

            # Step 5 — Generate answer
            answer = self._generate_answer(prompt)

            # Layer 2 + 3 — Hallucination check
            is_grounded, reason = self._check_hallucination(request.question, answer, top_chunks)
            if not is_grounded:
                logger.warning("Hallucination detected — reason: %s", reason)
                answer = "I found some information but cannot verify it's accurate enough to share. Please consult the source documents directly."

            sources = list({chunk.metadata.get("source_file", "") for chunk in top_chunks})
            return QueryResponse(
                answer=answer,
                sources=sources,
                chunks_used=len(top_chunks),
                hallucination_check=reason if not is_grounded else "grounded",
            )
        except RateLimitError as e:
            logger.error("Groq rate limit exceeded after retries: %s", e)
            raise
        except Exception as e:
            logger.error("Pipeline failed for question '%s': %s", request.question, e)
            raise

    def _generate_answer(self, prompt: str) -> str:
        response = self._call_groq_with_retry(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content

    # ── Streaming Pipeline ────────────────────────────────────────────────────

    def ask_stream(self, request: QueryRequest):
        """
        Fast streaming pipeline — skips multi-query for low TTFT.
        For nutrition (high stakes) → verify first, then stream.
        For travel/food (low stakes) → stream immediately.
        """
        logger.info("Stream query: %s", request.question)

        try:
            category = [request.category] if request.category else self._detect_categories(request.question)
            chunk_type = self._detect_chunk_type(request.question)
            is_high_stakes = bool(category and any(c in HIGH_STAKES_CATEGORIES for c in category))

            # Single query — no multi-query for fast TTFT
            candidates = self._search(
                [request.question],
                top_k=settings.rerank_candidates,
                category=category,
                chunk_type=chunk_type,
            )

            if not candidates:
                yield "data: I could not find relevant information to answer your question.\n\n"
                yield "data: [DONE]\n\n"
                return

            top_chunks = self._rerank(request.question, candidates, top_k=request.top_k)

            if not self._check_retrieval_confidence(top_chunks):
                yield "data: I don't have enough information in my knowledge base to answer this question.\n\n"
                yield "data: [DONE]\n\n"
                return

            prompt = self._build_prompt(request.question, top_chunks)

            if is_high_stakes:
                # High stakes — verify first, then stream verified answer
                logger.info("High stakes query — verifying before streaming")
                answer = self._generate_answer(prompt)
                is_grounded, reason = self._check_hallucination(request.question, answer, top_chunks)

                if not is_grounded:
                    yield "data: I found some information but cannot verify its accuracy. Please consult source documents directly.\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Stream verified answer token by token
                for word in answer.split(" "):
                    yield f"data: {word} \n\n"

            else:
                # Low stakes — stream immediately
                try:
                    response = self._call_groq_with_retry(
                        model=settings.groq_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.3,
                        max_tokens=1024,
                        stream=True,
                    )
                    for chunk in response:
                        token = chunk.choices[0].delta.content
                        if token:
                            yield f"data: {token}\n\n"
                except Exception as e:
                    logger.error("Streaming generation failed: %s", e)
                    yield "data: An error occurred while generating the response. Please try again.\n\n"
                    yield "data: [DONE]\n\n"
                    return

            sources = list({chunk.metadata.get("source_file", "") for chunk in top_chunks})
            yield f"data: [SOURCES]{','.join(sources)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error("Stream pipeline failed: %s", e)
            yield "data: An internal error occurred. Please try again.\n\n"
            yield "data: [DONE]\n\n"


# ── Singleton ─────────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


# ── Router ────────────────────────────────────────────────────────────────────

@router.post("/ask", response_model=QueryResponse)
async def ask(request: QueryRequest):
    """Full RAG pipeline — multi-query + hybrid + rerank + hallucination check."""
    try:
        pipeline = get_pipeline()
        return pipeline.ask(request)
    except RateLimitError:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later.",
        )
    except Exception as e:
        logger.error("Unhandled error in /ask: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.post("/ask/stream")
async def ask_stream(request: QueryRequest):
    """
    Streaming RAG pipeline — fast TTFT, no multi-query.
    High stakes (nutrition) → verify first then stream.
    Low stakes (travel/food) → stream immediately.
    """
    try:
        pipeline = get_pipeline()
        return StreamingResponse(
            pipeline.ask_stream(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",   # disable nginx buffering
            },
        )
    except Exception as e:
        logger.error("Unhandled error in /ask/stream: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error.")


@router.get("/health")
async def query_health():
    return {"status": "ok", "model": settings.groq_model}
