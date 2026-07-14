import logging
import math
from typing import Annotated
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
from src.settings import settings
from src.api.routers.query import get_pipeline
from src.agentic.long_term_memory import get_memory_store

logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """
    State flows through every node in the graph.
    Persisted to checkpointer after every node execution.

    messages:          full conversation history (auto-appended via add_messages)
    awaiting_approval: True when graph is paused at human_approval_node
    """
    messages: Annotated[list, add_messages]
    awaiting_approval: bool
    user_id: str


# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def rag_search(query: str, category: str | None = None) -> str:
    """
    Search Yatra.ai knowledge base for travel, nutrition, and food information.
    Use for questions about Indian destinations, dietary guidelines, and recipes.

    Args:
        query: search query string
        category: one of 'travel', 'nutrition', 'food', or None to search all
    """
    try:
        pipeline = get_pipeline()
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

        logger.info("rag_search: '%s' category=%s → %d chunks", query, category, len(top_chunks))
        return results

    except Exception as e:
        logger.error("rag_search failed: %s", e)
        return f"Search failed: {str(e)}"


@tool
def calculator(expression: str) -> str:
    """
    Perform mathematical calculations for budgets, calories, distances.
    Only calculate what is explicitly asked — do not assume extra details.

    Args:
        expression: mathematical expression e.g. '500 * 3' or '5000 / 3'
    """
    try:
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


TOOLS = [rag_search, calculator]

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are Yatra.ai, an intelligent Indian travel and food planning agent.
You help users plan trips, find healthy food options, and get nutritional advice for Indian cuisine.

You have access to these tools:
- rag_search: search knowledge base for travel, food, nutrition information
- calculator: perform mathematical calculations

RULES:
- Use calculator for ANY numerical calculations BEFORE searching
- Never repeat the same tool with same input twice
- If category-specific search returns irrelevant results, retry with category: null
- CRITICAL: Only answer what user explicitly asked — never assume extra details
- CRITICAL: Once you have a calculation result, use it directly — do not recalculate
- If after 3 searches you cannot find specific information, answer with what you have
- Always give a complete, helpful final answer
"""

# ── Graph Nodes ───────────────────────────────────────────────────────────────

from groq import Groq
import json

def llm_node(state: AgentState) -> AgentState:
    """
    LLM node — calls Groq with full message history from state.
    Long-term memory facts injected into system prompt here.
    """
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.1,
    ).bind_tools(TOOLS, parallel_tool_calls=False,tool_choice="auto")

    messages = state["messages"]

    if not any(isinstance(m, SystemMessage) for m in messages):
        user_id = state.get("user_id", "default")
        memory_facts = get_memory_store().format_for_prompt(user_id)
        system_with_memory = SYSTEM_PROMPT + memory_facts
        messages = [SystemMessage(content=system_with_memory)] + messages

    response = llm.invoke(messages)
    logger.info(
        "LLM response — tool_calls: %d",
        len(response.tool_calls) if hasattr(response, "tool_calls") else 0,
    )
    
    logger.info("LLM answer content: '%s'", response.content)
    return {
        "messages": [response],
        "awaiting_approval": False,
    }

def human_approval_node(state: AgentState) -> AgentState:
    """
    Human-in-the-loop node using LangGraph interrupt().

    How it works:
      1. interrupt() suspends the graph at this exact point
      2. Graph state is saved to checkpointer
      3. Caller receives control back with awaiting_approval=True
      4. Human calls /lg/approve endpoint
      5. graph.invoke(Command(resume=value)) resumes from this point
      6. human_input receives the resume value ("yes" or "no")

    This is the correct LangGraph HITL pattern — not polling, not websockets.
    The graph is truly suspended and resumed via checkpointer.
    """
    logger.info("Graph pausing — awaiting human approval via interrupt()")

    # interrupt() suspends graph here — returns resume value when continued
    human_input = interrupt(
        "⚠️ Nutrition/dietary advice detected. Approve this query? Reply yes/no"
    )

    if str(human_input).lower().strip() != "yes":
        logger.info("Human rejected — cancelling")
        return {
            "messages": [AIMessage(
                content="Request cancelled as per your instruction. "
                        "Feel free to ask something else."
            )],
            "awaiting_approval": False,
        }

    logger.info("Human approved — continuing")
    return {"awaiting_approval": False}


# ── Conditional Edges ─────────────────────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """
    Router — decides next node after LLM responds.

    Returns:
        "human_approval" → nutrition query detected → pause for human
        "tools"          → LLM wants to call a tool → execute tools
        "end"            → LLM gave final answer → END
    """
    last_message = state["messages"][-1]

    if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        logger.info("Routing to END — no tool calls")
        return "end"

    tool_names = [tc["name"] for tc in last_message.tool_calls]
    tool_inputs = [str(tc.get("args", {})) for tc in last_message.tool_calls]

    # Check if nutrition-related query — requires human approval
    nutrition_keywords = [
    "calorie", "protein", "vitamin", "diabetic", 
    "dietary guidelines", "nutrition facts"
]
    is_nutrition_query = (
        "rag_search" in tool_names
        and any(
            keyword in " ".join(tool_inputs).lower()
            for keyword in nutrition_keywords
        )
    )

    # Skip approval for personal memory lookups
    personal_keywords = ["gokul", "my name", "my preference", "about me"]
    is_personal_lookup = any(k in " ".join(tool_inputs).lower() for k in personal_keywords)

    if is_nutrition_query and not is_personal_lookup:
        return "human_approval"
    if is_nutrition_query:
        logger.info("Nutrition query — routing to human_approval")
        return "human_approval"

    logger.info("Routing to tools: %s", tool_names)
    return "tools"


def after_approval(state: AgentState) -> str:
    """
    Router after human_approval_node.
    If still awaiting → END (human rejected or graph still paused).
    If approved → continue to tools.
    """
    if state.get("awaiting_approval", False):
        return "end"
    return "tools"


# ── Build Graph ───────────────────────────────────────────────────────────────

def build_graph(checkpointer=None) -> StateGraph:
    """
    Build LangGraph agent with:
      - Checkpointing    → MemorySaver persists state after every node
      - Short-term memory → same thread_id loads previous state
      - Human-in-the-loop → interrupt() pauses before nutrition queries
      - Conditional routing → should_continue() decides flow

    Graph structure:
        START → llm → [router] ──► human_approval ──► tools → llm
                               ──► tools → llm
                               ──► END
    """
    graph = StateGraph(AgentState)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("human_approval", human_approval_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "human_approval": "human_approval",
            "end": END,
        },
    )

    graph.add_conditional_edges(
        "human_approval",
        after_approval,
        {
            "tools": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "llm")

    return graph.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────────────────────────────

# MemorySaver — in-memory checkpointer
# Production replacement → AsyncPostgresSaver from langgraph-checkpoint-postgres
_checkpointer = MemorySaver()
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=_checkpointer)
        logger.info("LangGraph agent compiled with MemorySaver checkpointer")
    return _graph


# ── Run ───────────────────────────────────────────────────────────────────────

def run_graph_agent(question: str, thread_id: str = "default") -> dict:
    """
    Run LangGraph agent with checkpointing and short-term memory.

    Short-term memory via checkpointing:
      - First call with thread_id="user_123" → state saved to MemorySaver
      - Second call with thread_id="user_123" → state loaded → agent remembers
      - Different thread_id → fresh conversation, no memory

    If graph pauses at human_approval_node:
      - Returns awaiting_approval=True
      - Caller must POST to /agent/lg/approve to continue
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
    "messages": [HumanMessage(content=question)],
    "user_id": thread_id,
    "iteration_count": 0,
    "failed_agents": [],
    "agents_needed": [],    # ← missing
    "is_complete": False,   # ← missing
}

    logger.info("LangGraph agent starting — thread_id: %s", thread_id)

    try:
        final_state = graph.invoke(initial_state, config=config)
        # Extract facts from conversation and save to long-term memory
        conversation = [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", 
            "content": m.content}
            for m in final_state["messages"]
            if hasattr(m, "content") and m.content
    ]
        get_memory_store().extract_and_save(thread_id, conversation)
    except Exception as e:
        logger.error("Graph execution error: %s", e)
        return {"answer": f"Agent error: {str(e)}", "thread_id": thread_id}

    final_message = final_state["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    steps = []
    for msg in final_state["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                steps.append({
                    "tool": tc["name"],
                    "input": tc["args"],
                })

    awaiting = final_state.get("awaiting_approval", False)

    logger.info(
        "LangGraph complete — thread_id: %s, steps: %d, awaiting_approval: %s",
        thread_id, len(steps), awaiting,
    )

    return {
        "answer": answer,
        "steps": steps,
        "total_messages": len(final_state["messages"]),
        "thread_id": thread_id,
        "awaiting_approval": awaiting,
    }


def resume_after_approval(thread_id: str, approved: bool) -> dict:
    """
    Resume graph after human approval decision.

    Uses Command(resume=value) — the correct LangGraph resumption pattern.
    Graph loads state from checkpointer for thread_id and continues
    from the exact interrupt() point in human_approval_node.

    approved=True  → human_input receives "yes" → graph continues to tools
    approved=False → human_input receives "no"  → graph cancels gracefully
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    resume_value = "yes" if approved else "no"
    logger.info("Resuming graph — thread_id: %s, approved: %s", thread_id, approved)

    try:
        # Command(resume=value) resumes from interrupt() point
        final_state = graph.invoke(
            Command(resume=resume_value),
            config=config,
        )
    except Exception as e:
        logger.error("Resume error: %s", e)
        return {"answer": f"Resume error: {str(e)}", "thread_id": thread_id}

    final_message = final_state["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    return {
        "answer": answer,
        "thread_id": thread_id,
        "awaiting_approval": False,
    }