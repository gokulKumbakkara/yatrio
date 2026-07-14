import logging
from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from src.ingestion.embedder import EmbeddedChunk, get_embedding_function
from src.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """Unified result from any retrieval strategy."""
    text: str
    metadata: dict
    score: float = 0.0


class VectorDBLoader:
    """
    Manages ChromaDB for writing and retrieval.
    Supports three retrieval strategies:
      - similarity: basic cosine similarity
      - mmr: maximal marginal relevance (diverse results)
      - hybrid: BM25 + vector search fused with RRF
    """

    def __init__(self):
        self.db = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=get_embedding_function(),
            persist_directory=settings.chroma_persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB initialised — collection: %s, existing docs: %d",
            settings.chroma_collection_name,
            self.db._collection.count(),
        )

        # Build BM25 index on startup from existing chunks
        self._bm25: BM25Okapi | None = None
        self._bm25_texts: list[str] = []
        self._bm25_metadatas: list[dict] = []
        self._build_bm25_index()

    # ── Write ──────────────────────────────────────────────────────────────────

    def load(self, embedded_chunks: list[EmbeddedChunk]) -> None:
        """Load embedded chunks into ChromaDB. Idempotent via upsert."""
        if not embedded_chunks:
            logger.warning("No chunks to load")
            return

        batch_size = 100
        total = len(embedded_chunks)

        for i in range(0, total, batch_size):
            batch = embedded_chunks[i: i + batch_size]
            try:
                self.db._collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    embeddings=[chunk.embedding for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[
                        {
                            "category": chunk.category,
                            "source_file": chunk.source_file,
                            "chunk_index": chunk.chunk_index,
                            "chunk_type": chunk.chunk_type,
                            "embedding_model": chunk.metadata.get("embedding_model", ""),
                            "num_pages": chunk.metadata.get("num_pages", 0),
                            "source_path": chunk.metadata.get("source_path", ""),
                        }
                        for chunk in batch
                    ],
                )
                logger.info("Loaded batch %d/%d", min(i + batch_size, total), total)
            except Exception as e:
                logger.error(
                    "Failed to upsert batch %d-%d: %s", i, min(i + batch_size, total), e
                )
                raise

        logger.info(
            "Load complete — total chunks in DB: %d",
            self.db._collection.count(),
        )

        # Rebuild BM25 index after new chunks are loaded
        self._build_bm25_index()

    # ── BM25 Index ─────────────────────────────────────────────────────────────

    def _build_bm25_index(self) -> None:
        """
        Fetch all chunk texts from ChromaDB and build in-memory BM25 index.
        Called on startup and after every load().
        """
        try:
            count = self.db._collection.count()
            if count == 0:
                logger.info("No chunks in DB — skipping BM25 index build")
                return

            logger.info("Building BM25 index from %d chunks...", count)

            results = self.db._collection.get(include=["documents", "metadatas"])

            self._bm25_texts = results["documents"]
            self._bm25_metadatas = results["metadatas"]

            tokenized_corpus = [text.lower().split() for text in self._bm25_texts]
            self._bm25 = BM25Okapi(tokenized_corpus)

            logger.info("BM25 index built — %d documents indexed", len(self._bm25_texts))
        except Exception as e:
            logger.error("Failed to build BM25 index: %s", e)
            self._bm25 = None
            self._bm25_texts = []
            self._bm25_metadatas = []

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def vector_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
        strategy: str = "mmr",
    ) -> list[Document]:
        """Vector search using MMR or similarity."""
        try:
            search_kwargs = {"k": top_k}
            if filters:
                search_kwargs["filter"] = filters
            if strategy == "mmr":
                search_kwargs["fetch_k"] = top_k * 4

            retriever = self.db.as_retriever(
                search_type=strategy,
                search_kwargs=search_kwargs,
            )
            return retriever.invoke(query)
        except Exception as e:
            logger.error("Vector search failed for query '%s': %s", query[:50], e)
            return []

    def bm25_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        BM25 keyword search over all chunk texts.
        Applies metadata filter manually after scoring.
        """
        if not self._bm25:
            logger.warning("BM25 index not built — returning empty results")
            return []

        try:
            query_tokens = query.lower().split()
            scores = self._bm25.get_scores(query_tokens)

            scored = list(zip(scores, self._bm25_texts, self._bm25_metadatas))

            if filters:
                scored = self._apply_filter(scored, filters)

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]

            return [
                RetrievedChunk(text=text, metadata=meta, score=score)
                for score, text, meta in top
            ]
        except Exception as e:
            logger.error("BM25 search failed for query '%s': %s", query[:50], e)
            return []

    def hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Hybrid search — BM25 + vector (MMR) fused with RRF.
        Returns top_k chunks ranked by combined relevance.
        """
        try:
            vector_docs = self.vector_search(query, top_k, filters, strategy="mmr")
            bm25_docs = self.bm25_search(query, top_k, filters)

            vector_chunks = [
                RetrievedChunk(text=doc.page_content, metadata=doc.metadata)
                for doc in vector_docs
            ]

            return self._rrf_fusion(vector_chunks, bm25_docs, top_k)
        except Exception as e:
            logger.error("Hybrid search failed for query '%s': %s", query[:50], e)
            return []

    def _rrf_fusion(
        self,
        vector_results: list[RetrievedChunk],
        bm25_results: list[RetrievedChunk],
        top_k: int,
        k: int = 60,  # RRF constant — 60 is standard
    ) -> list[RetrievedChunk]:
        """
        Reciprocal Rank Fusion — merges two ranked lists into one.
        RRF score = 1/(rank + k) summed across both lists.
        Chunks appearing in both lists get a natural boost.
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        # Score vector results
        for rank, chunk in enumerate(vector_results):
            key = chunk.text[:100]  # use text snippet as key
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + k)
            chunk_map[key] = chunk

        # Score BM25 results
        for rank, chunk in enumerate(bm25_results):
            key = chunk.text[:100]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1 / (rank + k)
            chunk_map[key] = chunk

        # Sort by combined RRF score
        sorted_keys = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            chunk = chunk_map[key]
            chunk.score = rrf_scores[key]
            results.append(chunk)

        return results

    def _apply_filter(
        self,
        scored: list[tuple],
        filters: dict,
    ) -> list[tuple]:
        """Apply metadata filters to BM25 results manually."""
        if "$and" in filters:
            conditions = filters["$and"]
            return [
                (score, text, meta)
                for score, text, meta in scored
                if all(meta.get(k) == v for cond in conditions for k, v in cond.items())
            ]
        elif "$or" in filters:
            conditions = filters["$or"]
            return [
                (score, text, meta)
                for score, text, meta in scored
                if any(meta.get(k) == v for cond in conditions for k, v in cond.items())
            ]
        else:
            return [
                (score, text, meta)
                for score, text, meta in scored
                if all(meta.get(k) == v for k, v in filters.items())
            ]

    def get_stats(self) -> dict:
        count = self.db._collection.count()
        return {
            "collection": settings.chroma_collection_name,
            "total_chunks": count,
            "persist_dir": settings.chroma_persist_dir,
            "bm25_indexed": len(self._bm25_texts),
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_loader_instance: VectorDBLoader | None = None


def get_loader() -> VectorDBLoader:
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = VectorDBLoader()
    return _loader_instance


# ── Orchestrator-facing function ──────────────────────────────────────────────

def load_to_vectordb(embedded_chunks: list[EmbeddedChunk]) -> None:
    loader = get_loader()
    loader.load(embedded_chunks)
