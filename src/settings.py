from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Parser ────────────────────────────────────────────────────────
    parser_type: str = "docling"
    raw_dir: str = "data/raw"
    parsed_dir: str = "data/parsed"

    # ── Embedder ──────────────────────────────────────────────────────
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32

    # ── Vector DB ─────────────────────────────────────────────────────
    chroma_collection_name: str = "yatra_ai"
    chroma_persist_dir: str = "data/vectordb"

    # ── Query ─────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama-3.3-70b-versatile"
    top_k: int = 5

    #Retrieval
    retrieval_strategy: str = "mmr" 

    reranking_model_id: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    rerank_top_k: int = 20      # fetch this many, then rerank
    final_top_k: int = 5        # send this many to LLM after reranking
    RERANKING_CROSS_ENCODER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-4-v2"
    memory_db_url: str = "sqlite:///data/memory.db"

# Single instance — import this everywhere
settings = Settings()