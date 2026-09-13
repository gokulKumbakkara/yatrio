# Yatrio (fit_yatra.ai)

An intelligent Indian travel & food companion — a RAG-powered API that answers questions about destinations, dietary guidelines, and recipes using an agentic pipeline with tool use.

## Features

- **RAG search** over an ingested knowledge base of Indian travel, nutrition, and food content (categories: travel / nutrition / food)
- **Agentic query answering** — a LangGraph-based agent that can call `rag_search`, run a `calculator` tool for budget/nutrition math, and produce a `final_answer`
- **Document ingestion pipeline** — loading, cleaning, chunking, and embedding source documents (via Docling + sentence-transformers)
- **Hybrid retrieval** — vector search (ChromaDB) combined with BM25 keyword search
- **RAG evaluation** — automated answer-quality evaluation via Ragas
- **Long-term memory** for the agent across conversations

## Tech Stack

| Layer | Tech |
|---|---|
| API | FastAPI, Uvicorn |
| Agent orchestration | LangGraph, LangChain (`langchain-core`, `langchain-groq`) |
| LLM | Groq (`groq` SDK) |
| Retrieval | ChromaDB (vector), `rank-bm25` (keyword), `langchain-huggingface` + `sentence-transformers` (embeddings) |
| Document parsing | Docling |
| Evaluation | Ragas |
| Storage | SQLAlchemy + `aiosqlite` |

## Getting Started

### Prerequisites

- Python 3.12 or 3.13
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- A Groq API key

### Installation

```bash
git clone https://github.com/gokulKumbakkara/yatrio
cd yatrio

uv sync
cp .env.example .env  # if present, otherwise create one with your API keys

uv run main.py
```

The API starts on `http://localhost:8000`. Health check: `GET /health`.

### API surface

| Prefix | Purpose |
|---|---|
| `/ingest` | Document ingestion into the knowledge base |
| `/query` | Direct RAG query endpoint |
| `/evaluate` | Ragas-based answer evaluation |
| `/agent` | Agentic query answering (tool-using LangGraph agent) |

## Project Structure

```
yatrio/
├── main.py                  # uvicorn entrypoint
└── src/
    ├── api/
    │   ├── app.py            # FastAPI app + router registration
    │   └── routers/          # ingestion, query, evaluation, agent
    ├── agentic/               # agent, tools, prompts, long-term memory
    ├── ingestion/              # loader, cleaner, chunker, embedder
    ├── evaluation/             # Ragas-based evaluator
    └── settings.py
```
