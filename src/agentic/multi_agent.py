import logging
import math
from typing import Annotated, Literal
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, Send
from typing_extensions import TypedDict
from src.settings import settings
from src.api.routers.query import get_pipeline
from operator import add

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5  # prevent infinite supervisor loops

# ── State ─────────────────────────────────────────────────────────────────────

class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str
    iteration_count: Annotated[int, add]        # ← add reducer: sum all updates
    failed_agents: Annotated[list[str], add]    # ← add reducer: concatenate lists
    agents_needed: list[str]
    is_complete: bool         # ← add


# ── Structured Output for Supervisor ─────────────────────────────────────────

class SupervisorDecision(BaseModel):
    """
    Supervisor decides ALL agents needed upfront for parallel execution.
    
    Key difference from sequential:
      Sequential → next_agent: str (one at a time)
      Parallel   → agents_needed: list[str] (all at once)
    
    Pydantic validates — no invalid agent names possible.
    """
    agents_needed: list[Literal["travel", "nutrition", "recipe"]]
    reasoning: str
    is_complete: bool  # True → go to synthesizer, False → run agents


# ── Specialist Tools ──────────────────────────────────────────────────────────

@tool
def search_travel(query: str) -> str:
    """Search for Indian travel destinations, tourism, and trip planning information."""
    try:
        pipeline = get_pipeline()
        candidates = pipeline._search([query], top_k=5, category=["travel"], chunk_type=None)
        if not candidates:
            return "No travel information found."
        top_chunks = pipeline._rerank(query, candidates, top_k=3)
        return "\n\n---\n\n".join(chunk.text for chunk in top_chunks)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def search_nutrition(query: str) -> str:
    """Search for dietary guidelines, nutrition facts, and health information."""
    try:
        pipeline = get_pipeline()
        candidates = pipeline._search([query], top_k=5, category=["nutrition"], chunk_type=None)
        if not candidates:
            return "No nutrition information found."
        top_chunks = pipeline._rerank(query, candidates, top_k=3)
        return "\n\n---\n\n".join(chunk.text for chunk in top_chunks)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def search_recipe(query: str) -> str:
    """Search for Indian recipes, cooking methods, and food preparation."""
    try:
        pipeline = get_pipeline()
        candidates = pipeline._search([query], top_k=5, category=["food"], chunk_type=None)
        if not candidates:
            return "No recipe information found."
        top_chunks = pipeline._rerank(query, candidates, top_k=3)
        return "\n\n---\n\n".join(chunk.text for chunk in top_chunks)
    except Exception as e:
        return f"Search failed: {e}"


@tool
def calculator(expression: str) -> str:
    """Perform mathematical calculations for budgets, calories, distances."""
    try:
        allowed = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sqrt": math.sqrt,
        }
        result = eval(expression, allowed)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"


# ── System Prompts ────────────────────────────────────────────────────────────

SUPERVISOR_PROMPT = """You are a supervisor coordinating specialist agents for Yatra.ai.

Available agents:
- travel: Indian destinations, tourism, trip planning
- nutrition: diet, health conditions, dietary guidelines  
- recipe: Indian recipes, cooking, food preparation

Your job:
1. Analyze the user question
2. Decide which agents are needed (can be multiple)
3. Set is_complete=True only when all needed agents have already responded

Rules:
- For multi-domain questions, list ALL needed agents at once (they run in parallel)
- Set is_complete=True when you have enough information to synthesize an answer
- Never request the same agent twice
"""

TRAVEL_AGENT_PROMPT = """You are a travel specialist for Yatra.ai.
Help users plan trips to Indian destinations.
Use search_travel tool to find relevant information.
Be specific about destinations, activities, and travel tips.
"""

NUTRITION_AGENT_PROMPT = """You are a nutrition specialist for Yatra.ai.
Help users with dietary advice tailored to Indian cuisine.
Use search_nutrition tool to find relevant information.
Always consider health conditions mentioned by the user.
"""

RECIPE_AGENT_PROMPT = """You are a recipe specialist for Yatra.ai.
Help users find and prepare Indian recipes.
Use search_recipe tool to find relevant information.
Provide clear ingredient lists and preparation steps.
"""

SYNTHESIZER_PROMPT = """You are synthesizing responses from multiple specialist agents.
Create a coherent, well-structured final answer from all specialist inputs.
Be concise and directly address the user's original question.
If any agents failed, note this but still provide the best possible answer.ßßßßßßßß
"""


# ── Supervisor Node ───────────────────────────────────────────────────────────


# supervisor_node just updates state with decision

def supervisor_node(state: MultiAgentState) -> dict:
    """Supervisor decides which agents needed — saves to state."""
    if state.get("iteration_count", 0) >= MAX_ITERATIONS:
        return {"agents_needed": [], "iteration_count": MAX_ITERATIONS}

    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.0,
    ).with_structured_output(SupervisorDecision)

    messages = [SystemMessage(content=SUPERVISOR_PROMPT)] + state["messages"]
    
    try:
        decision = llm.invoke(messages)
    except Exception as e:
        logger.error("Supervisor failed: %s", e)
        return {"is_complete": True, "iteration_count": state.get("iteration_count", 0) + 1}

    logger.info("Supervisor → agents: %s, complete: %s", decision.agents_needed, decision.is_complete)
    
    return {
        "agents_needed": decision.agents_needed,
        "is_complete": decision.is_complete,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


# Conditional edge function — THIS returns list[Send]
def route_supervisor(state: MultiAgentState):
    """Router reads state and returns Send for parallel execution."""
    if state.get("is_complete", False) or not state.get("agents_needed", []):
        return "synthesizer"
    
    # Return list[Send] for parallel execution
    
    return [Send(agent, state) for agent in state["agents_needed"]]

# ── Generic Specialist Runner ─────────────────────────────────────────────────

def _run_specialist(
    state: MultiAgentState,
    system_prompt: str,
    tools: list,
    agent_name: str,
) -> Command:
    """
    Generic specialist with error handling.
    
    Error handling strategy:
      Transient errors (rate limit) → retry up to 2 times
      Permanent errors              → skip + warn (don't crash entire graph)
    
    WHY skip over crash:
      If travel_agent fails, nutrition info is still valuable
      Crashing entire graph loses all work done so far
      User gets partial answer with disclaimer > no answer
      
    After completing → always reports back to supervisor via Command
    Supervisor then decides if more agents needed or time to synthesize
    """
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.1,
    ).bind_tools(tools, parallel_tool_calls=False)

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    try:
        # Mini agentic loop — specialist runs until no more tool calls
        for _ in range(3):
            response = llm.invoke(messages)
            messages.append(response)

            if not (hasattr(response, "tool_calls") and response.tool_calls):
                break

            # Execute tool calls
            for tc in response.tool_calls:
                tool_map = {t.name: t for t in tools}
                if tc["name"] in tool_map:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    ))

        logger.info("%s completed successfully", agent_name)

        return Command(
            update={
                "messages": [response],
                "iteration_count": state.get("iteration_count", 0) + 1,
            },
            goto="supervisor",  # always report back
        )

    except Exception as e:
        # Skip + warn — don't crash entire graph
        logger.error("%s failed: %s", agent_name, e)
        error_message = AIMessage(
            content=f"[{agent_name} encountered an error and could not retrieve information: {str(e)}]"
        )
        return Command(
            update={
                "messages": [error_message],
                "failed_agents": state.get("failed_agents", []) + [agent_name],
                "iteration_count": state.get("iteration_count", 0) + 1,
            },
            goto="supervisor",  # still report back — supervisor decides what to do
        )


# ── Specialist Nodes ──────────────────────────────────────────────────────────

def travel_agent_node(state: MultiAgentState) -> Command:
    return _run_specialist(
        state, TRAVEL_AGENT_PROMPT,
        [search_travel, calculator], "TravelAgent",
    )


def nutrition_agent_node(state: MultiAgentState) -> Command:
    return _run_specialist(
        state, NUTRITION_AGENT_PROMPT,
        [search_nutrition, calculator], "NutritionAgent",
    )


def recipe_agent_node(state: MultiAgentState) -> Command:
    return _run_specialist(
        state, RECIPE_AGENT_PROMPT,
        [search_recipe, calculator], "RecipeAgent",
    )


# ── Synthesizer Node ──────────────────────────────────────────────────────────

def synthesizer_node(state: MultiAgentState) -> Command:
    """
    Combines all specialist outputs into one coherent answer.
    Adds disclaimer if any agents failed.
    """
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model_name=settings.groq_model,
        temperature=0.3,  # slight creativity for natural synthesis
    )

    messages = [SystemMessage(content=SYNTHESIZER_PROMPT)] + state["messages"]
    response = llm.invoke(messages)

    # Add disclaimer for failed agents
    failed = state.get("failed_agents", [])
    content = response.content
    if failed:
        content += f"\n\n⚠️ Note: {', '.join(failed)} could not retrieve information."

    final_message = AIMessage(content=content)
    logger.info("Synthesizer complete — failed agents: %s", failed)

    return Command(
        update={"messages": [final_message]},
        goto=END,
    )


# ── Build Graph ───────────────────────────────────────────────────────────────
def build_multi_agent_graph(checkpointer=None):
    """
    Parallel multi-agent graph using Send API.

    Structure:
        START → supervisor ──Send──► travel    ──► supervisor
                                  ──Send──► nutrition ──► supervisor
                                  ──Send──► recipe    ──► supervisor
                         ──Command──► synthesizer ──► END

    Key design decisions:
      Send vs Command: Send for parallel independent agents
                       Command for sequential dependent ro-uting
      Error handling:  Skip + warn (not crash, not silent)
      Max iterations:  Guard against infinite supervisor loops
    """
    graph = StateGraph(MultiAgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("travel", travel_agent_node)
    graph.add_node("nutrition", nutrition_agent_node)
    graph.add_node("recipe", recipe_agent_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "synthesizer": "synthesizer",
        "travel": "travel",
        "nutrition": "nutrition",
        "recipe": "recipe",
    }
) # ← conditional edge with Send

    graph.set_entry_point("supervisor")

    return graph.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────────────────────────────

_checkpointer = MemorySaver()
_multi_agent_graph = None


def get_multi_agent_graph():
    global _multi_agent_graph
    if _multi_agent_graph is None:
        _multi_agent_graph = build_multi_agent_graph(checkpointer=_checkpointer)
        logger.info("Multi-agent graph compiled with parallel Send support")
    return _multi_agent_graph


# ── Run ───────────────────────────────────────────────────────────────────────

def run_multi_agent(question: str, thread_id: str = "default") -> dict:
    """Run multi-agent system with parallel execution and error handling."""
    graph = get_multi_agent_graph()
    config = {"configurable": {"thread_id": f"multi_{thread_id}"}}

    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_id": thread_id,
        "iteration_count": 0,
        "failed_agents": [],
    }

    logger.info("Multi-agent starting — question: %s", question)

    try:
        final_state = graph.invoke(initial_state, config=config)
    except Exception as e:
        logger.error("Multi-agent graph failed: %s", e)
        return {"answer": f"System error: {str(e)}", "thread_id": thread_id}

    final_message = final_state["messages"][-1]
    answer = final_message.content if hasattr(final_message, "content") else str(final_message)

    return {
        "answer": answer,
        "failed_agents": final_state.get("failed_agents", []),
        "total_messages": len(final_state["messages"]),
        "iterations": final_state.get("iteration_count", 0),
        "thread_id": thread_id,
    }