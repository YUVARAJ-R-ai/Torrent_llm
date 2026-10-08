"""Structured agent roles for RecursiveMAS (Issue #11).

Defines Planner, Solver, and Critic roles with structured prompt templates,
system instructions, and step contracts for multi-agent reasoning.
"""

from __future__ import annotations

import enum
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


class RoleType(enum.StrEnum):
    """Canonical agent role types in RecursiveMAS."""

    PLANNER = "planner"
    SOLVER = "solver"
    CRITIC = "critic"


@dataclass
class PlanStep:
    """A discrete sub-task produced by the Planner."""

    index: int
    description: str


@dataclass
class PlanResult:
    """Structured output contract from the Planner role."""

    task: str
    steps: list[str]
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def step_count(self) -> int:
        return len(self.steps)


@dataclass
class SolveResult:
    """Structured output contract from the Solver role."""

    plan: list[str]
    solution: str
    raw_text: str
    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueResult:
    """Structured output contract from the Critic role."""

    approved: bool
    feedback: str
    score: float
    raw_text: str
    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


PLANNER_SYSTEM_PROMPT = (
    "You are the Planner agent in a distributed multi-agent system (RecursiveMAS). "
    "Your objective is to decompose the given high-level task into a clear, numbered sequence "
    "of actionable, logically ordered sub-tasks for the Solver to execute.\n\n"
    "Output your plan in the following format:\n"
    "Step 1: <first sub-task>\n"
    "Step 2: <second sub-task>\n"
    "Step 3: <third sub-task>\n"
    "..."
)

SOLVER_SYSTEM_PROMPT = (
    "You are the Solver agent in a distributed multi-agent system (RecursiveMAS). "
    "Your objective is to execute the sub-tasks defined in the plan, providing step-by-step "
    "reasoning and a concrete, comprehensive final solution. "
    "If critique feedback is provided from a previous round, address all requested revisions."
)

CRITIC_SYSTEM_PROMPT = (
    "You are the Critic agent in a distributed multi-agent system (RecursiveMAS). "
    "Your objective is to critically evaluate the candidate solution against the task "
    "requirements and decomposition plan. Review correctness, logical consistency, "
    "technical soundness, and edge case coverage.\n\n"
    "You must format your response strictly as follows:\n"
    "STATUS: APPROVED (or STATUS: REVISE)\n"
    "SCORE: <float between 0.0 and 1.0>\n"
    "FEEDBACK: <detailed critique, strengths, and specific refinement instructions>"
)


class AgentRole(ABC):
    """Abstract base class for RecursiveMAS agent roles."""

    role_type: RoleType
    system_prompt: str

    def __init__(self, system_prompt: str | None = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt

    @abstractmethod
    def format_prompt(self, **kwargs: Any) -> str:
        """Format input prompt for this agent role."""

    @abstractmethod
    def parse_output(self, raw_text: str, **kwargs: Any) -> Any:
        """Parse raw model output into role's structured contract."""


class PlannerRole(AgentRole):
    """Planner role: decomposes goals into structured step-by-step sub-tasks."""

    role_type = RoleType.PLANNER
    system_prompt = PLANNER_SYSTEM_PROMPT

    def format_prompt(self, task: str, context: str | None = None) -> str:
        """Format the planning prompt."""
        prompt = (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[TASK]\n{task}\n"
        )
        if context:
            prompt += f"\n[CONTEXT]\n{context}\n"
        prompt += "\n[PLAN]\n"
        return prompt

    def parse_output(self, raw_text: str, task: str = "") -> PlanResult:
        """Parse raw output into structured PlanResult."""
        steps: list[str] = []
        step_pattern = re.compile(
            r"^(?:Step\s*\d+[:.]|\d+[\.)]|[-*])\s*(.+)$",
            re.IGNORECASE | re.MULTILINE,
        )
        for match in step_pattern.finditer(raw_text):
            step_desc = match.group(1).strip()
            if step_desc:
                steps.append(step_desc)

        # Fallback if no formatted steps were detected
        if not steps:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            non_empty = [ln for ln in lines if not ln.startswith(("[", "Plan:", "PLAN"))]
            if non_empty:
                steps = non_empty[:5]
            else:
                steps = [raw_text.strip() or "Execute primary task."]

        return PlanResult(task=task, steps=steps, raw_text=raw_text)


class SolverRole(AgentRole):
    """Solver role: executes sub-tasks and produces candidate/refined solutions."""

    role_type = RoleType.SOLVER
    system_prompt = SOLVER_SYSTEM_PROMPT

    def format_prompt(
        self,
        task: str,
        plan: list[str],
        previous_solution: str | None = None,
        feedback: str | None = None,
        round_index: int = 0,
    ) -> str:
        """Format the solving prompt, supporting iterative refinement."""
        formatted_plan = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(plan))
        prompt = (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[TASK]\n{task}\n\n"
            f"[DECOMPOSED PLAN]\n{formatted_plan}\n"
        )
        if previous_solution and feedback:
            prompt += (
                f"\n[PREVIOUS ATTEMPT (ROUND {round_index})]\n{previous_solution}\n\n"
                f"[CRITIQUE FEEDBACK]\n{feedback}\n\n"
                "Please provide a revised, improved solution that directly resolves all "
                "critique points.\n"
            )
        prompt += "\n[SOLUTION]\n"
        return prompt

    def parse_output(
        self,
        raw_text: str,
        plan: list[str] | None = None,
        round_index: int = 0,
    ) -> SolveResult:
        """Parse raw output into structured SolveResult."""
        cleaned = raw_text.strip()
        # Remove leading marker if present
        if cleaned.upper().startswith("[SOLUTION]"):
            cleaned = cleaned[len("[SOLUTION]") :].strip()
        return SolveResult(
            plan=plan or [],
            solution=cleaned,
            raw_text=raw_text,
            round_index=round_index,
        )


class CriticRole(AgentRole):
    """Critic role: reviews consistency, correctness, and approves or requests refinement."""

    role_type = RoleType.CRITIC
    system_prompt = CRITIC_SYSTEM_PROMPT

    def format_prompt(
        self,
        task: str,
        plan: list[str],
        solution: str,
        round_index: int = 0,
    ) -> str:
        """Format the critique prompt."""
        formatted_plan = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(plan))
        return (
            f"[SYSTEM]\n{self.system_prompt}\n\n"
            f"[TASK]\n{task}\n\n"
            f"[PLAN]\n{formatted_plan}\n\n"
            f"[CANDIDATE SOLUTION (ROUND {round_index})]\n{solution}\n\n"
            "[EVALUATION]\n"
        )

    def parse_output(self, raw_text: str, round_index: int = 0) -> CritiqueResult:
        """Parse raw output into structured CritiqueResult."""
        approved = False
        score = 0.5
        feedback = ""

        status_match = re.search(
            r"STATUS:\s*(APPROVED|REVISE|REJECT|PENDING)",
            raw_text,
            re.IGNORECASE,
        )
        if status_match:
            val = status_match.group(1).upper()
            approved = val == "APPROVED"
        else:
            # Check APPROVED: YES / NO
            app_match = re.search(
                r"APPROVED:\s*(YES|NO|TRUE|FALSE)",
                raw_text,
                re.IGNORECASE,
            )
            if app_match:
                approved = app_match.group(1).upper() in ("YES", "TRUE")
            else:
                # Textual heuristic
                lower_text = raw_text.lower()
                pos_signals = ["approved", "looks good", "correct", "well-executed", "acceptable"]
                neg_signals = ["revise", "needs work", "issue", "incorrect", "flaw", "missing"]
                pos_count = sum(1 for s in pos_signals if s in lower_text)
                neg_count = sum(1 for s in neg_signals if s in lower_text)
                approved = pos_count > neg_count

        score_match = re.search(r"SCORE:\s*([0-9]*\.?[0-9]+)", raw_text, re.IGNORECASE)
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.9 if approved else 0.4
        else:
            score = 0.9 if approved else 0.4

        feedback_match = re.search(
            r"FEEDBACK:\s*(.+)",
            raw_text,
            re.IGNORECASE | re.DOTALL,
        )
        if feedback_match:
            feedback = feedback_match.group(1).strip()
        else:
            # Strip status/score lines if present, use remainder as feedback
            lines = [
                ln.strip()
                for ln in raw_text.splitlines()
                if not re.match(r"^(?:STATUS|SCORE|APPROVED):", ln.strip(), re.IGNORECASE)
            ]
            fallback_msg = "Approved." if approved else "Revisions requested."
            feedback = "\n".join(ln for ln in lines if ln) or fallback_msg

        return CritiqueResult(
            approved=approved,
            feedback=feedback,
            score=score,
            raw_text=raw_text,
            round_index=round_index,
        )
