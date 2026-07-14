import logging
from fastapi import APIRouter
from pydantic import BaseModel
from src.agentic.agent import get_agent
from src.agentic.langgraph_agent import run_graph_agent

from src.agentic.langgraph_agent import run_graph_agent, resume_after_approval

router = APIRouter()
logger = logging.getLogger(__name__)


class AgentRequest(BaseModel):
    question: str
    thread_id: str = "default" 

class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool


class AgentStepResponse(BaseModel):
    thought: str
    action: str
    action_input: str
    observation: str


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStepResponse]
    iterations: int
    stopped_reason: str


@router.post("/ask", response_model=AgentResponse)
async def agent_ask(request: AgentRequest):
    """
    ReAct agent endpoint.
    Agent decides which tools to use and iterates until it has an answer.
    """
    agent = get_agent()
    result = agent.run(request.question)

    return AgentResponse(
        answer=result.answer,
        steps=[
            AgentStepResponse(
                thought=step.thought,
                action=step.action,
                action_input=step.action_input,
                observation=step.observation,
            )
            for step in result.steps
        ],
        iterations=result.iterations,
        stopped_reason=result.stopped_reason,
    )


@router.get("/health")
async def agent_health():
    return {"status": "ok", "max_iterations": 5}

@router.post("/lg/ask")
async def langgraph_ask(request: AgentRequest):
    result = run_graph_agent(request.question, request.thread_id)  # ← add thread_id
    return result

@router.post("/lg/approve")
async def langgraph_approve(request: ApprovalRequest):
    """Resume graph after human approval."""
    result = resume_after_approval(request.thread_id, request.approved)
    return result

from src.agentic.multi_agent import run_multi_agent

class MultiAgentRequest(BaseModel):
    question: str
    thread_id: str = "default"

@router.post("/multi/ask")
async def multi_agent_ask(request: MultiAgentRequest):
    """
    Multi-agent endpoint — supervisor routes to specialists in parallel.
    Returns answer, failed_agents, iterations for observability.
    """
    result = run_multi_agent(request.question, request.thread_id)
    return result