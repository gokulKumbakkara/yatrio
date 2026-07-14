import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.ingestion.cleaner import CleanedDocument


@dataclass
class Chunk:
    chunk_id: str
    text: str
    category: str
    source_file: str
    chunk_index: int
    chunk_type: str        # "text" or "table"
    metadata: dict = field(default_factory=dict)


# ── Base ─────────────────────────────────────────────────────────────────────

class BaseChunker(ABC):
    """
    Abstract chunker — all category chunkers inherit from this.
    Handles both text and table chunking.
    Text chunking is delegated to subclasses via _get_splitter().
    Table chunking is handled here — same logic for all categories.
    """

    def chunk(self, doc: CleanedDocument) -> list[Chunk]:
        chunks = []

        # ── Text chunks ───────────────────────────────────────────────
        text_chunks = self._chunk_text(doc)
        chunks.extend(text_chunks)

        # ── Table chunks ──────────────────────────────────────────────
        table_chunks = self._chunk_tables(doc)
        chunks.extend(table_chunks)

        # Re-index all chunks sequentially
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        return chunks

    def _chunk_text(self, doc: CleanedDocument) -> list[Chunk]:
        splitter = self._get_splitter()
        splits = splitter.split_text(doc.cleaned_text)

        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                text=split,
                category=doc.category,
                source_file=doc.file_name,
                chunk_index=0,  # re-indexed later
                chunk_type="text",
                metadata={
                    **doc.metadata,
                    "source_path": doc.source_path,
                },
            )
            for split in splits
            if len(split.strip()) >= 50  # skip empty splits
        ]

    def _chunk_tables(self, doc: CleanedDocument) -> list[Chunk]:
        """
        Each table row becomes a chunk with headers prepended.
        If a row text exceeds chunk size, falls back to RecursiveCharacterTextSplitter.
        """
        chunks = []
        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
        )

        for table in doc.tables:
            rows = table.get("data", [])
            caption = table.get("caption", "")

            if not rows:
                continue

            headers = list(rows[0].keys())

            for row in rows:
                # Format: "Header1: value1 | Header2: value2 | ..."
                row_text = " | ".join(
                    f"{col}: {row.get(col, '')}" for col in headers
                )
                if caption:
                    row_text = f"{caption} — {row_text}"

                # Fallback if row text exceeds chunk size
                if len(row_text) > 512:
                    splits = fallback_splitter.split_text(row_text)
                else:
                    splits = [row_text]

                for split in splits:
                    chunks.append(
                        Chunk(
                            chunk_id=str(uuid.uuid4()),
                            text=split,
                            category=doc.category,
                            source_file=doc.file_name,
                            chunk_index=0,  # re-indexed later
                            chunk_type="table",
                            metadata={
                                **doc.metadata,
                                "source_path": doc.source_path,
                                "table_caption": caption,
                                "num_rows": table.get("num_rows", 0),
                                "num_cols": table.get("num_cols", 0),
                            },
                        )
                    )

        return chunks

    @abstractmethod
    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        """Each category defines its own splitter config."""
        pass


# ── Category Chunkers ─────────────────────────────────────────────────────────

class TravelChunker(BaseChunker):
    """
    Destination India and travel guides.
    Medium chunks — mixed content of descriptions, tips, itineraries.
    Split on sections and paragraphs.
    """

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=384,
            chunk_overlap=64,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )


class NutritionChunker(BaseChunker):
    """
    Dietary Guidelines PDF.
    Larger chunks — nutritional context needs more surrounding text to be useful.
    Split on chapters and sections first.
    """

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=100,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )


class FoodChunker(BaseChunker):
    """
    Recipe books — Manjula's Kitchen, Recipe Book.
    Smaller chunks — recipes are naturally short and self-contained.
    Split on recipe boundaries.
    """

    def _get_splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=256,
            chunk_overlap=50,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )


# ── Factory ───────────────────────────────────────────────────────────────────

class ChunkerFactory:
    """
    Returns the right chunker based on document category.
    """

    _registry: dict[str, type[BaseChunker]] = {
        "travel": TravelChunker,
        "nutrition": NutritionChunker,
        "food": FoodChunker,
    }

    @classmethod
    def get_chunker(cls, category: str) -> BaseChunker:
        chunker_class = cls._registry.get(category.lower())
        if not chunker_class:
            raise ValueError(
                f"No chunker registered for category '{category}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return chunker_class()

    @classmethod
    def register(cls, category: str, chunker_class: type[BaseChunker]) -> None:
        """Register a new category chunker at runtime."""
        cls._registry[category] = chunker_class


# ── Orchestrator-facing function ──────────────────────────────────────────────

def chunk_documents(docs: list[CleanedDocument]) -> list[Chunk]:
    """
    Called by the ingestion orchestrator.
    Dispatches each document to the right chunker via factory.
    """
    all_chunks = []
    for doc in docs:
        chunker = ChunkerFactory.get_chunker(doc.category)
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        print(f"  {doc.file_name} → {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    return all_chunks