import json
import logging
import math
from src.ingestion.loader import get_loader
from src.api.routers.query import get_pipeline

logger = logging.getLogger(__name__)


# ── Tool Definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "rag_search",
        "description": (
            "Search Yatra.ai knowledge base for travel, nutrition, and food information. "
            "Use for questions about Indian destinations, dietary guidelines, and recipes. "
            "Input: {\"query\": \"search query\", \"category\": \"travel|nutrition|food|null\"}"
        ),
    },
    {
        "name": "calculator",
        "description": (
            "Perform mathematical calculations. "
            "Use for budget calculations, nutritional math, distance estimates. "
            "Input: {\"expression\": \"5000 / 3\"}"
        ),
    },
    {
        "name": "final_answer",
        "description": (
            "Use this when you have enough information to answer the user's question. "
            "Input: {\"answer\": \"your complete answer\"}"
        ),
    },
]


# ── Tool Implementations ──────────────────────────────────────────────────────

def rag_search(query: str, category: str | None = None) -> str:
    """
    Search ChromaDB using hybrid search + reranking.
    Returns top chunks as formatted string.
    """
    try:
        loader = get_loader()
        pipeline = get_pipeline()

        # Use single query search (no multi-query — agent handles iteration)
        category_list = [category] if category else None
        candidates = pipeline._search(
            [query],
            top_k=5,
            category=category_list,
            chunk_type=None,
        )

        if not candidates:
            return "No relevant information found for this query."

        top_chunks = pipeline._rerank(query, candidates, top_k=3)
        results = "\n\n---\n\n".join(chunk.text for chunk in top_chunks)

        logger.info("rag_search: query='%s' category='%s' → %d chunks", query, category, len(top_chunks))
        return results

    except Exception as e:
        logger.error("rag_search failed: %s", e)
        return f"Search failed: {str(e)}"


def calculator(expression: str) -> str:
    """
    Safely evaluate mathematical expressions.
    Supports: +, -, *, /, **, (, )
    """
    try:
        # Safe eval — only allow math operations
        allowed = {
            "__builtins__": {},
            "abs": abs, "round": round,
            "min": min, "max": max,
            "sqrt": math.sqrt, "ceil": math.ceil, "floor": math.floor,
        }
        result = eval(expression, allowed)  # noqa: S307
        logger.info("calculator: '%s' = %s", expression, result)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


# ── Tool Registry ─────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    "rag_search": lambda inputs: rag_search(**inputs),
    "calculator": lambda inputs: calculator(**inputs),
}


def execute_tool(tool_name: str, tool_input: str) -> str:
    """
    Parse tool input JSON and execute the tool.
    Returns observation string.
    """
    if tool_name == "final_answer":
        # final_answer is handled by the agent loop, not executed here
        return ""

    if tool_name not in TOOL_REGISTRY:
        return f"Unknown tool: '{tool_name}'. Available tools: {list(TOOL_REGISTRY.keys())}"

    try: 
        clean_input = tool_input.split("#")[0].strip()  # ← add this
        inputs = json.loads(clean_input)
        return TOOL_REGISTRY[tool_name](inputs)
    except json.JSONDecodeError:
        return f"Invalid JSON input for tool '{tool_name}': {tool_input}"
    except Exception as e:
        return f"Tool '{tool_name}' failed: {str(e)}"