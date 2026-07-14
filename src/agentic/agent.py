import re
import logging
from dataclasses import dataclass, field
from groq import Groq
from src.settings import settings
from src.agentic.tools import TOOLS, execute_tool
from src.agentic.prompts import (
    REACT_SYSTEM_PROMPT,
    REACT_HUMAN_PROMPT,
    format_tools_for_prompt,
)

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


@dataclass
class AgentStep:
    """Single ReAct step — Thought + Action + Observation."""
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class AgentResult:
    """Final result from the agent."""
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = ""  # "final_answer" | "max_iterations" | "error"


class ReActAgent:
    """
    ReAct agent built from scratch.
    Loop: Thought → Action → Observation → repeat until Final Answer.

    This is the foundation that LangGraph automates.
    Understanding this loop makes LangGraph intuitive.
    """

    def __init__(self):
        self.groq = Groq(api_key=settings.groq_api_key)
        self.tools_desc, self.tool_names = format_tools_for_prompt(TOOLS)
        logger.info("ReAct agent initialised with tools: %s", self.tool_names)

    def _build_system_prompt(self) -> str:
        return REACT_SYSTEM_PROMPT.format(
            tools=self.tools_desc,
            tool_names=self.tool_names,
        )

    def _build_history(self, steps: list[AgentStep]) -> str:
        """Format previous steps as history for the next LLM call."""
        if not steps:
            return ""

        history = ""
        for step in steps:
            history += f"Thought: {step.thought}\n"
            history += f"Action: {step.action}\n"
            history += f"Action Input: {step.action_input}\n"
            history += f"Observation: {step.observation}\n\n"
        return history

    def _call_llm(self, question: str, steps: list[AgentStep]) -> str:
        """Call Groq LLM with current state — returns raw LLM output."""
        history = self._build_history(steps)
        user_prompt = REACT_HUMAN_PROMPT.format(
            question=question,
            history=history,
        )

        response = self.groq.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,   # low temp — must follow format strictly
            max_tokens=1024,
            stop=["Observation:"],  # stop before writing observation itself
        )
        return response.choices[0].message.content.strip()

    def _parse_llm_output(self, output: str) -> tuple[str, str, str, bool]:
        """
        Parse LLM output into (thought, action, action_input, is_final).
        Returns is_final=True if LLM wrote Final Answer.
        """
        # Check for Final Answer
        final_answer_match = re.search(
            r"Final Answer:\s*(.+?)$", output, re.DOTALL | re.IGNORECASE
        )
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            thought_match = re.search(r"Thought:\s*(.+?)(?=Final Answer:)", output, re.DOTALL)
            thought = thought_match.group(1).strip() if thought_match else ""
            return thought, "final_answer", answer, True

        # Parse Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:)", output, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else output

        # Parse Action
        action_match = re.search(r"Action:\s*(.+?)(?=Action Input:)", output, re.DOTALL)
        action = action_match.group(1).strip() if action_match else ""

        # Parse Action Input
        action_input_match = re.search(r"Action Input:\s*(.+?)$", output, re.DOTALL)
        action_input = action_input_match.group(1).strip() if action_input_match else "{}"

        return thought, action, action_input, False

    def run(self, question: str) -> AgentResult:
        """
        Main ReAct loop.
        Thought → Action → Observation → repeat until Final Answer or max iterations.
        """
        logger.info("Agent starting — question: %s", question)
        steps: list[AgentStep] = []

        for iteration in range(MAX_ITERATIONS):
            logger.info("Iteration %d/%d", iteration + 1, MAX_ITERATIONS)

            # Call LLM
            llm_output = self._call_llm(question, steps)
            logger.debug("LLM output:\n%s", llm_output)

            # Parse output
            thought, action, action_input, is_final = self._parse_llm_output(llm_output)

            # Final answer reached
            if is_final:
                logger.info("Agent reached Final Answer after %d iterations", iteration + 1)
                return AgentResult(
                    answer=action_input,  # action_input holds the answer for final_answer
                    steps=steps,
                    iterations=iteration + 1,
                    stopped_reason="final_answer",
                )

            # Execute tool
            logger.info("Executing tool: %s with input: %s", action, action_input)
            observation = execute_tool(action, action_input)
            if observation == steps[-1].observation if steps else None:
                observation = "Same result as before. Try a different query or category."
            logger.info("Observation: %s", observation[:100])

            # Record step
            steps.append(AgentStep(
                thought=thought,
                action=action,
                action_input=action_input,
                observation=observation,
            ))

        # Max iterations reached
        logger.warning("Max iterations reached — returning best available answer")
        last_observation = steps[-1].observation if steps else "No information found"
        return AgentResult(
            answer=f"Based on available information: {last_observation}",
            steps=steps,
            iterations=MAX_ITERATIONS,
            stopped_reason="max_iterations",
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent: ReActAgent | None = None


def get_agent() -> ReActAgent:
    global _agent
    if _agent is None:
        _agent = ReActAgent()
    return _agent