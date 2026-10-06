"""RecursiveMAS multi-agent orchestrator across sharded layers (Issue #11).

Coordinates Planner -> Solver -> Critic roles with iterative refinement,
latent state bridging via RecursiveLink, and end-to-end wire bandwidth profiling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from torrent_llm.agent.link import RecursiveLink
from torrent_llm.agent.roles import (
    CriticRole,
    CritiqueResult,
    PlannerRole,
    PlanResult,
    SolveResult,
    SolverRole,
)

logger = logging.getLogger(__name__)


@dataclass
class StepLog:
    """Detailed log record of a single role execution step."""

    step_number: int
    role: str
    round_index: int
    content: str
    latency_s: float
    wire_bytes: int
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class MASRunResult:
    """Execution summary of an end-to-end multi-agent reasoning session."""

    task: str
    final_solution: str
    is_approved: bool
    rounds_executed: int
    total_wall_time_s: float
    per_agent_latency_s: dict[str, float]
    wire_bytes_transferred: int
    raw_bytes_equivalent: int
    compression_ratio: float
    savings_pct: float
    plan: PlanResult
    solution_history: list[SolveResult]
    critique_history: list[CritiqueResult]
    step_logs: list[StepLog] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Convert result into a JSON-serializable dictionary."""
        return {
            "task": self.task,
            "final_solution": self.final_solution,
            "is_approved": self.is_approved,
            "rounds_executed": self.rounds_executed,
            "total_wall_time_s": self.total_wall_time_s,
            "per_agent_latency_s": self.per_agent_latency_s,
            "wire_bytes_transferred": self.wire_bytes_transferred,
            "raw_bytes_equivalent": self.raw_bytes_equivalent,
            "compression_ratio": self.compression_ratio,
            "savings_pct": self.savings_pct,
            "plan": {
                "task": self.plan.task,
                "steps": self.plan.steps,
                "raw_text": self.plan.raw_text,
            },
            "solution_history": [asdict(s) for s in self.solution_history],
            "critique_history": [asdict(c) for c in self.critique_history],
            "step_logs": [asdict(log_entry) for log_entry in self.step_logs],
            "metadata": self.metadata,
        }

    def format_summary(self) -> str:
        """Format an ASCII report summarizing multi-agent resolution and bandwidth savings."""
        wire_kb = self.wire_bytes_transferred / 1024.0
        raw_kb = self.raw_bytes_equivalent / 1024.0

        lines = [
            "=" * 78,
            "RECURSIVEMAS MULTI-AGENT EXECUTION SUMMARY",
            "=" * 78,
            f"Task: {self.task}",
            f"Status: {'APPROVED' if self.is_approved else 'MAX ROUNDS REACHED'}",
            f"Rounds: {self.rounds_executed} | Wall Time: {self.total_wall_time_s:.3f}s",
            "-" * 78,
            "FINAL SOLUTION:",
            self.final_solution,
            "-" * 78,
            "WIRE BANDWIDTH & LATENCY METRICS:",
            (
                f"  Total Wire Bytes Transferred:   {self.wire_bytes_transferred:,} bytes "
                f"({wire_kb:.2f} KiB)"
            ),
            (
                f"  Raw Baseline Equivalent:        {self.raw_bytes_equivalent:,} bytes "
                f"({raw_kb:.2f} KiB)"
            ),
            f"  Wire Bandwidth Savings:         {self.savings_pct:.2f}%",
            f"  Wire Compression Ratio:         {self.compression_ratio:.2f}x",
            "  Per-Agent Latency:",
        ]
        for role, lat in self.per_agent_latency_s.items():
            lines.append(f"    - {role.capitalize()}: {lat:.3f}s")
        lines.append("=" * 78)
        return "\n".join(lines)


class RecursiveMASOrchestrator:
    """Multi-agent pipeline coordinating Planner -> Solver -> Critic.

    Executes role steps over the sharded network, bridging latent states through
    RecursiveLink, and managing the iterative critique and refinement loop.
    """

    def __init__(
        self,
        link: RecursiveLink,
        planner: PlannerRole | None = None,
        solver: SolverRole | None = None,
        critic: CriticRole | None = None,
        max_rounds: int = 3,
        max_new_tokens: int = 64,
    ) -> None:
        self.link = link
        self.planner = planner or PlannerRole()
        self.solver = solver or SolverRole()
        self.critic = critic or CriticRole()
        self.max_rounds = max_rounds
        self.max_new_tokens = max_new_tokens

    def run(self, task: str, context: str | None = None) -> MASRunResult:
        """Run the end-to-end multi-agent orchestration loop."""
        start_time = time.perf_counter()
        step_logs: list[StepLog] = []
        step_idx = 1

        logger.info("Starting RecursiveMAS run for task: %s", task)

        # -------------------------------------------------------------
        # Step 1: Planner Role (Decomposes task into structured plan)
        # -------------------------------------------------------------
        t0 = time.perf_counter()
        planner_prompt = self.planner.format_prompt(task, context=context)
        plan_text, _, planner_latent = self.link.step(
            role="planner",
            prompt=planner_prompt,
            max_new_tokens=self.max_new_tokens,
        )
        plan_lat = time.perf_counter() - t0
        plan_result = self.planner.parse_output(plan_text, task=task)

        step_logs.append(
            StepLog(
                step_number=step_idx,
                role="planner",
                round_index=0,
                content=plan_text,
                latency_s=plan_lat,
                wire_bytes=planner_latent.wire_bytes,
                details={"steps": plan_result.steps},
            )
        )
        step_idx += 1

        # -------------------------------------------------------------
        # Iterative Refinement Loop (Solver <-> Critic)
        # -------------------------------------------------------------
        solution_history: list[SolveResult] = []
        critique_history: list[CritiqueResult] = []

        current_latent = planner_latent
        current_feedback: str | None = None
        previous_solution: str | None = None
        last_solve_result: SolveResult | None = None
        last_critique_result: CritiqueResult | None = None

        rounds_run = 0
        for round_idx in range(self.max_rounds):
            rounds_run += 1
            # Solver Step
            t0 = time.perf_counter()
            solver_prompt = self.solver.format_prompt(
                task=task,
                plan=plan_result.steps,
                previous_solution=previous_solution,
                feedback=current_feedback,
                round_index=round_idx,
            )
            solve_text, _, solver_latent = self.link.step(
                role="solver",
                prompt=solver_prompt,
                max_new_tokens=self.max_new_tokens,
                prior_state=current_latent,
            )
            solve_lat = time.perf_counter() - t0
            last_solve_result = self.solver.parse_output(
                solve_text,
                plan=plan_result.steps,
                round_index=round_idx,
            )
            solution_history.append(last_solve_result)

            step_logs.append(
                StepLog(
                    step_number=step_idx,
                    role="solver",
                    round_index=round_idx,
                    content=solve_text,
                    latency_s=solve_lat,
                    wire_bytes=solver_latent.wire_bytes,
                    details={"round": round_idx},
                )
            )
            step_idx += 1

            # Critic Step
            t0 = time.perf_counter()
            critic_prompt = self.critic.format_prompt(
                task=task,
                plan=plan_result.steps,
                solution=last_solve_result.solution,
                round_index=round_idx,
            )
            critic_text, _, critic_latent = self.link.step(
                role="critic",
                prompt=critic_prompt,
                max_new_tokens=self.max_new_tokens,
                prior_state=solver_latent,
            )
            critic_lat = time.perf_counter() - t0
            last_critique_result = self.critic.parse_output(
                critic_text,
                round_index=round_idx,
            )
            critique_history.append(last_critique_result)

            step_logs.append(
                StepLog(
                    step_number=step_idx,
                    role="critic",
                    round_index=round_idx,
                    content=critic_text,
                    latency_s=critic_lat,
                    wire_bytes=critic_latent.wire_bytes,
                    details={
                        "round": round_idx,
                        "approved": last_critique_result.approved,
                        "score": last_critique_result.score,
                    },
                )
            )
            step_idx += 1

            if last_critique_result.approved:
                logger.info("Solution approved by Critic on round %d", round_idx)
                break

            # If not approved and more rounds remaining, prepare next refinement
            if round_idx < self.max_rounds - 1:
                logger.info("Refinement requested by Critic on round %d. Iterating.", round_idx)
                current_latent = critic_latent
                current_feedback = last_critique_result.feedback
                previous_solution = last_solve_result.solution

        total_wall_time = time.perf_counter() - start_time

        if last_solve_result is not None:
            final_solution = last_solve_result.solution
        else:
            final_solution = "No solution generated."

        is_approved = (
            last_critique_result.approved if last_critique_result is not None else False
        )

        return MASRunResult(
            task=task,
            final_solution=final_solution,
            is_approved=is_approved,
            rounds_executed=rounds_run,
            total_wall_time_s=total_wall_time,
            per_agent_latency_s=dict(self.link.metrics.per_role_latency_s),
            wire_bytes_transferred=self.link.metrics.total_wire_bytes,
            raw_bytes_equivalent=self.link.metrics.total_raw_bytes,
            compression_ratio=self.link.metrics.compression_ratio,
            savings_pct=self.link.metrics.savings_pct,
            plan=plan_result,
            solution_history=solution_history,
            critique_history=critique_history,
            step_logs=step_logs,
        )
