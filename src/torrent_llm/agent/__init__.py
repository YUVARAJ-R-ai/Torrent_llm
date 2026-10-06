"""Multi-agent latent-space collaboration across sharded layers (Issue #11)."""

from torrent_llm.agent.link import (
    AgentLatentState,
    LinkMetrics,
    RecursiveLink,
)
from torrent_llm.agent.orchestrator import (
    MASRunResult,
    RecursiveMASOrchestrator,
    StepLog,
)
from torrent_llm.agent.roles import (
    AgentRole,
    CriticRole,
    CritiqueResult,
    PlannerRole,
    PlanResult,
    PlanStep,
    RoleType,
    SolveResult,
    SolverRole,
)

__all__ = [
    "AgentLatentState",
    "AgentRole",
    "CritiqueResult",
    "CriticRole",
    "LinkMetrics",
    "MASRunResult",
    "PlanResult",
    "PlanStep",
    "PlannerRole",
    "RecursiveLink",
    "RecursiveMASOrchestrator",
    "RoleType",
    "SolveResult",
    "SolverRole",
    "StepLog",
]
