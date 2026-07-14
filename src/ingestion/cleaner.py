import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from src.ingestion.parser import ParsedDocument


@dataclass
class CleanedDocument:
    file_name: str
    category: str
    source_path: str
    cleaned_text: str
    tables: list[dict]
    metadata: dict


# ── Base ─────────────────────────────────────────────────────────────────────

class BaseCleaner(ABC):
    """
    Abstract cleaner — all category cleaners inherit from this.
    Defines the interface the orchestrator uses.
    """

    def clean(self, doc: ParsedDocument) -> CleanedDocument:
        text = doc.markdown_text
        text = self._remove_page_artifacts(text)
        text = self._remove_unicode_noise(text)
        text = self._normalize_whitespace(text)
        text = self._category_specific_clean(text)
        return CleanedDocument(
            file_name=doc.file_name,
            category=doc.category,
            source_path=doc.source_path,
            cleaned_text=text,
            tables=doc.tables,
            metadata=doc.metadata,
        )

    def _remove_page_artifacts(self, text: str) -> str:
        text = re.sub(r'\f', ' ', text)
        text = re.sub(r'\bPage\s+\d+\s+of\s+\d+\b', '', text)
        text = re.sub(r'\b\d+\s*/\s*\d+\b', '', text)
        text = re.sub(r'<!--\s*image\s*-->', '', text)           # ← add
        text = re.sub(r'[A-Z].*?\.{3,}.*?\d+', '', text, flags=re.MULTILINE)  # ← add
        return text
       

    def _remove_unicode_noise(self, text: str) -> str:
        """Remove OCR garbage and non-printable characters."""
        text = re.sub(r'[^\x00-\x7F\u0900-\u097F]+', ' ', text)  # keep ASCII + Devanagari
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text) # control chars
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple blank lines and trailing spaces."""
        text = re.sub(r' +', ' ', text)           # multiple spaces → single
        text = re.sub(r'\n{3,}', '\n\n', text)    # 3+ newlines → 2
        return text.strip()

    @abstractmethod
    def _category_specific_clean(self, text: str) -> str:
        """Each category cleaner implements its own specific rules."""
        pass


# ── Category Cleaners ─────────────────────────────────────────────────────────

class TravelCleaner(BaseCleaner):
    """
    Cleans Destination India and other travel PDFs.
    Removes watermarks, repeated branding headers, tourism boilerplate.
    """

    REPEATED_HEADERS = [
        r'Destination\s+India',
        r'Ministry\s+of\s+Tourism',
        r'Incredible\s+!ndia',
        r'Incredible\s+India',
    ]

    def _category_specific_clean(self, text: str) -> str:
        # Remove repeated branding that appears on every page
        for pattern in self.REPEATED_HEADERS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove standalone page numbers (common in travel guides)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        return text


class NutritionCleaner(BaseCleaner):
    """
    Cleans Dietary Guidelines PDF.
    Keeps chapter headings, nutrient names, table captions.
    Removes NIN branding headers, footnote markers.
    """

    REPEATED_HEADERS = [
        r'National\s+Institute\s+of\s+Nutrition',
        r'Dietary\s+Guidelines\s+for\s+Indians',
        r'ICMR[-\s]NIN',
    ]

    def _category_specific_clean(self, text: str) -> str:
        # Remove repeated institutional headers
        for pattern in self.REPEATED_HEADERS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove footnote markers like *, †, ‡ but keep the footnote text
        text = re.sub(r'(?<!\w)[*†‡§]+(?!\w)', '', text)

        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        return text


class FoodCleaner(BaseCleaner):
    """
    Cleans recipe book PDFs (Manjula's Kitchen, Recipe Book).
    Keeps recipe names, ingredients, method headings.
    Removes publisher boilerplate, copyright notices.
    """

    BOILERPLATE_PATTERNS = [
        r'All\s+rights\s+reserved',
        r'Copyright\s+©?\s*\d{4}',
        r'Printed\s+in\s+India',
        r'Published\s+by',
        r'ISBN[\s:\-]\d+',
    ]

    def _category_specific_clean(self, text: str) -> str:
        # Remove publisher boilerplate
        for pattern in self.BOILERPLATE_PATTERNS:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        # Remove standalone page numbers
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

        return text


# ── Factory ───────────────────────────────────────────────────────────────────

class CleanerFactory:
    """
    Returns the right cleaner based on document category.
    Orchestrator calls this — never instantiates cleaners directly.
    """

    _registry: dict[str, type[BaseCleaner]] = {
        "travel": TravelCleaner,
        "nutrition": NutritionCleaner,
        "food": FoodCleaner,
    }

    @classmethod
    def get_cleaner(cls, category: str) -> BaseCleaner:
        cleaner_class = cls._registry.get(category.lower())
        if not cleaner_class:
            raise ValueError(
                f"No cleaner registered for category '{category}'. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cleaner_class()

    @classmethod
    def register(cls, category: str, cleaner_class: type[BaseCleaner]) -> None:
        """Extend later — register a new category cleaner at runtime."""
        cls._registry[category] = cleaner_class


# ── Orchestrator-facing function ──────────────────────────────────────────────

def clean_documents(docs: list[ParsedDocument]) -> list[CleanedDocument]:
    """
    Called by the ingestion orchestrator.
    Dispatches each document to the right cleaner via factory.
    """
    cleaned = []
    for doc in docs:
        cleaner = CleanerFactory.get_cleaner(doc.category)
        cleaned_doc = cleaner.clean(doc)
        cleaned.append(cleaned_doc)
    return cleaned