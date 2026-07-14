import logging
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEmbeddings
from src.ingestion.chunker import Chunk
from src.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddedChunk:
    chunk_id: str
    text: str
    category: str
    source_file: str
    chunk_index: int
    chunk_type: str
    embedding: list[float]  # 384 floats
    metadata: dict = field(default_factory=dict)


class DocumentEmbedder:
    """
    Converts Chunk objects into EmbeddedChunk objects.
    Uses HuggingFaceEmbeddings (langchain wrapper over sentence-transformers).
    Processes in batches for performance.
    """

    def __init__(self):
        logger.info("Loading embedding model: %s", settings.embedding_model_id)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_id,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"batch_size": settings.batch_size},
        )
        logger.info("Embedding model loaded")

    def embed(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed all chunks in batches."""
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        logger.info("Embedding %d chunks...", len(chunks))
        try:
            vectors = self.embeddings.embed_documents(texts)
        except Exception as e:
            logger.error("Failed to embed %d chunks: %s", len(chunks), e)
            raise

        embedded_chunks = []
        for chunk, vector in zip(chunks, vectors):
            embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    category=chunk.category,
                    source_file=chunk.source_file,
                    chunk_index=chunk.chunk_index,
                    chunk_type=chunk.chunk_type,
                    embedding=vector,
                    metadata={
                        **chunk.metadata,
                        "embedding_model": settings.embedding_model_id,
                        "embedding_size": 384,
                    },
                )
            )

        logger.info("Embedding complete — %d chunks embedded", len(embedded_chunks))
        return embedded_chunks


# ── Singleton ─────────────────────────────────────────────────────────────────

_embedder_instance: DocumentEmbedder | None = None


def get_embedder() -> DocumentEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = DocumentEmbedder()
    return _embedder_instance


def get_embedding_function() -> HuggingFaceEmbeddings:
    """
    Returns the raw LangChain embedding function.
    Used by LangChain Chroma wrapper in loader.py and query.py.
    Reuses same model instance via singleton — no double loading.
    """
    return get_embedder().embeddings


# ── Orchestrator-facing function ──────────────────────────────────────────────

def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    embedder = get_embedder()
    return embedder.embed(chunks)