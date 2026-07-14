
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from src.settings import settings
# in __init__


logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    file_name: str
    category: str
    source_path: str
    markdown_text: str
    tables: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DocumentParser:
    """
    Walks data/raw/<category>/ folders and parses every PDF using Docling.
    Saves markdown + metadata JSON to data/parsed/<category>/.
    """

    def __init__(self):
        self.raw_dir = Path(settings.raw_dir)
        self.parsed_dir = Path(settings.parsed_dir)
        self.converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def parse_all(self) -> list[ParsedDocument]:
        """Parse all PDFs across all category folders."""
        parsed_docs = []

        for category_dir in sorted(self.raw_dir.iterdir()):
            if not category_dir.is_dir():
                continue

            category = category_dir.name
            pdfs = list(category_dir.glob("*.pdf"))
            logger.info("Category '%s': found %d PDF(s)", category, len(pdfs))

            for pdf_file in pdfs:
                doc = self._parse_single(pdf_file, category)
                if doc:
                    try:
                        self._save(doc)
                    except Exception as e:
                        logger.error("Failed to save parsed doc %s: %s", pdf_file.name, e)
                    parsed_docs.append(doc)

        logger.info("parse_all complete: %d document(s) parsed", len(parsed_docs))
        return parsed_docs

    def parse_category(self, category: str) -> list[ParsedDocument]:
        """Parse only PDFs under a specific category folder."""
        category_dir = self.raw_dir / category
        if not category_dir.exists():
            raise ValueError(f"Category folder not found: {category_dir}")

        parsed_docs = []
        for pdf_file in category_dir.glob("*.pdf"):
            doc = self._parse_single(pdf_file, category)
            if doc:
                self._save(doc)
                parsed_docs.append(doc)

        return parsed_docs

    def _parse_single(self, file_path: Path, category: str) -> Optional[ParsedDocument]:
        logger.info("Parsing: %s", file_path.name)
        try:
            result = self.converter.convert(str(file_path))
            doc = result.document

            markdown_text = doc.export_to_markdown()

            tables = []
            for table in doc.tables:
                try:
                    df = table.export_to_dataframe(doc)
                    tables.append({
                        "caption": table.caption_text(doc) if hasattr(table, "caption_text") else "",
                        "data": df.to_dict(orient="records"),
                        "num_rows": len(df),
                        "num_cols": len(df.columns),
                    })
                except Exception:
                    pass

            parsed = ParsedDocument(
                file_name=file_path.name,
                category=category,
                source_path=str(file_path),
                markdown_text=markdown_text,
                tables=tables,
                metadata={
                    "num_pages": len(doc.pages),
                    "num_tables": len(tables),
                    "num_characters": len(markdown_text),
                    "has_images": len(doc.pictures) > 0,
                },
            )
            logger.info(
                "Done: %s — %d page(s), %d table(s), %d chars",
                file_path.name,
                parsed.metadata["num_pages"],
                parsed.metadata["num_tables"],
                parsed.metadata["num_characters"],
            )
            return parsed

        except Exception as e:
            logger.error("Failed to parse %s: %s", file_path.name, e)
            return None

    def _save(self, doc: ParsedDocument) -> None:
        """Save markdown and metadata JSON to data/parsed/<category>/."""
        try:
            out_dir = self.parsed_dir / doc.category
            out_dir.mkdir(parents=True, exist_ok=True)

            stem = Path(doc.file_name).stem

            (out_dir / f"{stem}.md").write_text(doc.markdown_text, encoding="utf-8")

            (out_dir / f"{stem}_meta.json").write_text(
                json.dumps(
                    {
                        "file_name": doc.file_name,
                        "category": doc.category,
                        "source_path": doc.source_path,
                        "tables": doc.tables,
                        "metadata": doc.metadata,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception as e:
            logger.error("Failed to save parsed document %s: %s", doc.file_name, e)
            raise

    def is_already_parsed(self, file_name: str, category: str) -> bool:
        """Checkpoint check — skip parsing if markdown already exists."""
        stem = Path(file_name).stem
        return (self.parsed_dir / category / f"{stem}.md").exists()