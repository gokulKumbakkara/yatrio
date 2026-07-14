import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from groq import Groq
from src.settings import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

# ── Model ─────────────────────────────────────────────────────────────────────

class UserMemory(Base):
    """
    Key-value fact store per user.
    Flexible — any fact type without schema changes.

    Examples:
      user_id="user_gokul", fact_key="name",             fact_value="Gokul"
      user_id="user_gokul", fact_key="diet_preference",  fact_value="vegetarian"
      user_id="user_gokul", fact_key="health_condition", fact_value="diabetic"
      user_id="user_gokul", fact_key="location",         fact_value="Kerala"
    """
    __tablename__ = "user_memory"

    id         = Column(String, primary_key=True)  # user_id + fact_key
    user_id    = Column(String, nullable=False, index=True)
    fact_key   = Column(String, nullable=False)
    fact_value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# ── Database ──────────────────────────────────────────────────────────────────

class LongTermMemoryStore:
    """
    Persistent user memory store.
    Uses SQLite locally — swap DATABASE_URL for PostgreSQL in production.

    Two operations:
      extract_and_save() → end of conversation → LLM extracts facts → save to DB
      load_for_user()    → start of conversation → load facts → inject into prompt
    """

    def __init__(self):
        self.engine = create_engine(
            settings.memory_db_url,
            connect_args={"check_same_thread": False},  # SQLite specific
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("Long-term memory store initialised — %s", settings.memory_db_url)

    def _get_session(self) -> Session:
        return self.SessionLocal()

    # ── Extract facts from conversation ───────────────────────────────────────

    def extract_and_save(self, user_id: str, conversation: list[dict]) -> list[str]:
        """
        Use LLM to extract key facts from conversation.
        Save extracted facts to DB.
        Returns list of extracted fact strings for logging.

        Called at end of each agent conversation.
        """
        if not conversation:
            return []

        # Format conversation for LLM
        conv_text = "\n".join(
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in conversation
            if msg.get("content")
        )

        # Ask LLM to extract facts
        groq = Groq(api_key=settings.groq_api_key)
        prompt = f"""Extract key facts about the user from this conversation.
Only extract facts explicitly stated by the user.
Return ONLY facts in this exact format, one per line:
fact_key: fact_value

Valid fact keys: name, diet_preference, health_condition, location, cuisine_preference, budget_preference, travel_preference

Conversation:
{conv_text}

Facts (or empty if none found):"""

        response = groq.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,   # deterministic — fact extraction must be precise
            max_tokens=200,
        )

        facts_text = response.choices[0].message.content.strip()
        if not facts_text:
            return []

        # Parse and save facts
        saved_facts = []
        for line in facts_text.split("\n"):
            line = line.strip()
            if ":" not in line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            fact_key = parts[0].strip().lower().replace(" ", "_")
            fact_value = parts[1].strip()

            if fact_key and fact_value:
                self._upsert_fact(user_id, fact_key, fact_value)
                saved_facts.append(f"{fact_key}: {fact_value}")

        logger.info(
            "Extracted %d facts for user %s: %s",
            len(saved_facts), user_id, saved_facts,
        )
        return saved_facts

    def _upsert_fact(self, user_id: str, fact_key: str, fact_value: str) -> None:
        """Insert or update a fact — upsert pattern."""
        record_id = f"{user_id}:{fact_key}"

        with self._get_session() as session:
            # SQLite upsert — insert or replace on conflict
            stmt = sqlite_insert(UserMemory).values(
                id=record_id,
                user_id=user_id,
                fact_key=fact_key,
                fact_value=fact_value,
                updated_at=datetime.utcnow(),
            ).on_conflict_do_update(
                index_elements=["id"],
                set_={
                    "fact_value": fact_value,
                    "updated_at": datetime.utcnow(),
                },
            )
            session.execute(stmt)
            session.commit()

    # ── Load facts for user ───────────────────────────────────────────────────

    def load_for_user(self, user_id: str) -> dict[str, str]:
        """
        Load all facts for a user from DB.
        Returns dict of {fact_key: fact_value}.
        Called at start of each conversation.
        """
        with self._get_session() as session:
            facts = session.query(UserMemory).filter(
                UserMemory.user_id == user_id
            ).all()

        return {fact.fact_key: fact.fact_value for fact in facts}

    def format_for_prompt(self, user_id: str) -> str:
        """
        Format user facts as a string to inject into system prompt.
        Returns empty string if no facts stored.

        Example output:
          Known facts about this user:
          - name: Gokul
          - diet_preference: vegetarian
          - health_condition: diabetic
          - location: Kerala
        """
        facts = self.load_for_user(user_id)
        if not facts:
            return ""

        facts_str = "\n".join(f"  - {k}: {v}" for k, v in facts.items())
        return f"\nKnown facts about this user:\n{facts_str}\n"

    def clear_user_memory(self, user_id: str) -> None:
        """Delete all facts for a user — GDPR compliance."""
        with self._get_session() as session:
            session.query(UserMemory).filter(
                UserMemory.user_id == user_id
            ).delete()
            session.commit()
        logger.info("Cleared all memory for user: %s", user_id)


# ── Singleton ─────────────────────────────────────────────────────────────────

_memory_store: LongTermMemoryStore | None = None


def get_memory_store() -> LongTermMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = LongTermMemoryStore()
    return _memory_store