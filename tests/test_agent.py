"""Unit and integration tests for RecursiveMAS agent roles and RecursiveLink (Issue #11).

All tests run 100% offline in seconds using the random tiny model fixture from conftest.py,
validating role contracts, latent state compression, and sharded multi-agent orchestration.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from scripts.run_multiagent import build_arg_parser, run_multiagent

from torrent_llm.agent import (
    CriticRole,
    PlannerRole,
    RecursiveLink,
    RecursiveMASOrchestrator,
    SolverRole,
)
from torrent_llm.codec import get_codec
from torrent_llm.codec.lowrank import LowRankCodec
from torrent_llm.codec.raw import RawCodec
from torrent_llm.runner import ChainRunner, TopologyConfig
from torrent_llm.shard import ShardRuntime
from torrent_llm.transport import serve


@pytest.fixture
def served_agent_chain(model_factory, num_layers):
    """Spins up a 2-shard chain with LowRankCodec on dynamic ports for offline testing."""
    base = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "codec": {"name": "lowrank", "rank": 16},
            "nodes": [{"address": "127.0.0.1:0"}, {"address": "127.0.0.1:0"}],
        }
    )
    servers, addresses = [], []
    codec = get_codec("lowrank", rank=16)
    for spec in base.shard_plan():
        runtime = ShardRuntime(model_factory(), spec)
        server, port = serve(runtime, codec, host="127.0.0.1", model_id="tiny")
        servers.append(server)
        addresses.append(f"127.0.0.1:{port}")

    config = TopologyConfig.from_dict(
        {
            "model_id": "tiny",
            "num_layers": num_layers,
            "codec": {"name": "lowrank", "rank": 16},
            "nodes": [{"address": a} for a in addresses],
        }
    )
    yield config

    for server in servers:
        server.stop(grace=None)


# ==============================================================================
# Role Execution and Prompt Formatting Tests
# ==============================================================================


def test_planner_role_prompt_formatting():
    planner = PlannerRole()
    prompt = planner.format_prompt(
        task="Optimize inter-node activation passing",
        context="Network bandwidth is 1 Gbps",
    )
    assert "[SYSTEM]" in prompt
    assert "[TASK]" in prompt
    assert "Optimize inter-node activation passing" in prompt
    assert "[CONTEXT]" in prompt
    assert "Network bandwidth is 1 Gbps" in prompt
    assert "[PLAN]" in prompt


def test_planner_role_parse_output_structured():
    planner = PlannerRole()
    raw = (
        "Here is the plan:\n"
        "Step 1: Profile baseline activation size.\n"
        "Step 2: Apply low-rank projection.\n"
        "Step 3: Measure downstream perplexity.\n"
    )
    res = planner.parse_output(raw, task="Evaluate compression")
    assert res.task == "Evaluate compression"
    assert len(res.steps) == 3
    assert res.steps[0] == "Profile baseline activation size."
    assert res.steps[1] == "Apply low-rank projection."
    assert res.steps[2] == "Measure downstream perplexity."


def test_planner_role_parse_output_fallback():
    planner = PlannerRole()
    raw = "Partition layers equally and stream tokens sequentially."
    res = planner.parse_output(raw, task="Simple test")
    assert len(res.steps) >= 1
    assert "Partition layers" in res.steps[0]


def test_solver_role_prompt_formatting():
    solver = SolverRole()
    plan = ["Analyze hops", "Compress tensors"]
    prompt = solver.format_prompt(
        task="Solve sharding",
        plan=plan,
        previous_solution="Initial attempt",
        feedback="Include memory calculations",
        round_index=1,
    )
    assert "[TASK]" in prompt
    assert "Step 1: Analyze hops" in prompt
    assert "Step 2: Compress tensors" in prompt
    assert "[PREVIOUS ATTEMPT (ROUND 1)]" in prompt
    assert "Initial attempt" in prompt
    assert "[CRITIQUE FEEDBACK]" in prompt
    assert "Include memory calculations" in prompt


def test_solver_role_parse_output():
    solver = SolverRole()
    raw = "[SOLUTION]\nWe divide 28 layers into 2 shards of 14 layers each."
    res = solver.parse_output(raw, plan=["Divide layers"], round_index=0)
    assert "28 layers into 2 shards" in res.solution
    assert res.round_index == 0


def test_critic_role_prompt_formatting():
    critic = CriticRole()
    prompt = critic.format_prompt(
        task="Test task",
        plan=["Step 1"],
        solution="Proposed solution",
        round_index=0,
    )
    assert "[TASK]" in prompt
    assert "[PLAN]" in prompt
    assert "[CANDIDATE SOLUTION (ROUND 0)]" in prompt
    assert "[EVALUATION]" in prompt


def test_critic_role_parse_output_approved():
    critic = CriticRole()
    raw = (
        "STATUS: APPROVED\n"
        "SCORE: 0.95\n"
        "FEEDBACK: Solution is sound, covers all edge cases and boundary conditions."
    )
    res = critic.parse_output(raw, round_index=0)
    assert res.approved is True
    assert res.score == 0.95
    assert "covers all edge cases" in res.feedback


def test_critic_role_parse_output_revise():
    critic = CriticRole()
    raw = (
        "STATUS: REVISE\n"
        "SCORE: 0.40\n"
        "FEEDBACK: Sharding plan does not account for LM head placement on the tail node."
    )
    res = critic.parse_output(raw, round_index=1)
    assert res.approved is False
    assert res.score == 0.40
    assert "LM head placement" in res.feedback
    assert res.round_index == 1


def test_critic_role_parse_output_fallback():
    critic = CriticRole()
    raw = "The implementation looks completely correct and approved."
    res = critic.parse_output(raw)
    assert res.approved is True


# ==============================================================================
# RecursiveLink Encoding, Compression, and Latent Transfer Tests
# ==============================================================================


def test_recursive_link_encode_lowrank_compression():
    hidden_size = 64
    rank = 16
    codec = LowRankCodec(rank=rank)
    link = RecursiveLink(codec=codec, hidden_size=hidden_size)

    activation = torch.randn(1, 10, hidden_size)
    state = link.encode_latent(activation, role="planner")

    assert state.role == "planner"
    assert state.is_compressed is True
    assert state.tensor.shape == (1, 10, rank)
    assert state.wire_bytes < state.raw_bytes


def test_recursive_link_decode_shape_and_dtype():
    hidden_size = 64
    rank = 16
    codec = LowRankCodec(rank=rank)
    link = RecursiveLink(codec=codec, hidden_size=hidden_size)

    activation = torch.randn(1, 8, hidden_size, dtype=torch.float32)
    state = link.encode_latent(activation, role="solver")
    decoded = link.decode_latent(state)

    assert decoded.shape == activation.shape
    assert decoded.dtype == activation.dtype


def test_recursive_link_bridge_latent_transfer():
    link = RecursiveLink(codec=LowRankCodec(rank=16), hidden_size=64)
    activation = torch.randn(1, 12, 64)
    state = link.encode_latent(activation, role="planner")

    bridged = link.bridge("planner", "solver", state)

    assert bridged.role == "solver"
    assert link.metrics.latent_transfers == 1
    assert link.metrics.inter_agent_wire_bytes == state.wire_bytes
    assert link.metrics.inter_agent_raw_bytes == state.raw_bytes
    assert link.metrics.per_role_wire_bytes["planner"] == state.wire_bytes


def test_recursive_link_raw_vs_lowrank_wire_bytes():
    seq_len = 32
    hidden_size = 64
    rank = 16
    tensor = torch.randn(1, seq_len, hidden_size)

    raw_link = RecursiveLink(codec=RawCodec(), hidden_size=hidden_size)
    lr_link = RecursiveLink(codec=LowRankCodec(rank=rank), hidden_size=hidden_size)

    raw_state = raw_link.encode_latent(tensor, role="planner")
    lr_state = lr_link.encode_latent(tensor, role="planner")

    assert lr_state.wire_bytes < raw_state.wire_bytes
    # Low-rank wire bytes should be significantly less than raw
    ratio = raw_state.wire_bytes / lr_state.wire_bytes
    assert ratio > 2.0


def test_recursive_link_step_mock_execution():
    link = RecursiveLink(codec=LowRankCodec(rank=16), hidden_size=64)
    text, tokens, latent = link.step(
        role="planner",
        prompt="Decompose task",
        max_new_tokens=16,
    )

    assert isinstance(text, str)
    assert tokens.ndim >= 1
    assert latent.role == "planner"
    assert latent.is_compressed is True
    assert link.metrics.per_role_latency_s["planner"] > 0
    assert link.metrics.total_wire_bytes > 0


# ==============================================================================
# Multi-Agent Orchestration Loop Tests
# ==============================================================================


def test_orchestrator_single_round_approval():
    mock_responses = {
        "planner": ["Step 1: Partition. Step 2: Wire."],
        "solver": ["Solution: Partitioned into 2 shards."],
        "critic": ["STATUS: APPROVED\nSCORE: 0.95\nFEEDBACK: Perfect."],
    }
    link = RecursiveLink(
        codec=LowRankCodec(rank=16),
        hidden_size=64,
        mock_responses=mock_responses,
    )
    orchestrator = RecursiveMASOrchestrator(link, max_rounds=3)
    result = orchestrator.run(task="Shard model across 2 nodes")

    assert result.is_approved is True
    assert result.rounds_executed == 1
    assert len(result.solution_history) == 1
    assert len(result.critique_history) == 1
    assert "Partitioned into 2 shards" in result.final_solution
    assert result.wire_bytes_transferred > 0
    assert result.compression_ratio > 1.0


def test_orchestrator_iterative_refinement():
    mock_responses = {
        "planner": ["Step 1: Compute rank."],
        "solver": [
            "Attempt 1: Selected rank 64.",
            "Attempt 2: Selected rank 128 to improve perplexity.",
        ],
        "critic": [
            "STATUS: REVISE\nSCORE: 0.5\nFEEDBACK: Rank 64 causes too much loss.",
            "STATUS: APPROVED\nSCORE: 0.9\nFEEDBACK: Rank 128 matches target perplexity.",
        ],
    }
    link = RecursiveLink(
        codec=LowRankCodec(rank=16),
        hidden_size=64,
        mock_responses=mock_responses,
    )
    orchestrator = RecursiveMASOrchestrator(link, max_rounds=3)
    result = orchestrator.run(task="Select compression rank")

    assert result.is_approved is True
    assert result.rounds_executed == 2
    assert len(result.solution_history) == 2
    assert len(result.critique_history) == 2
    assert "Attempt 2" in result.final_solution


def test_orchestrator_max_rounds_reached():
    mock_responses = {
        "planner": ["Step 1: Infinite retry."],
        "solver": ["Attempt 1", "Attempt 2"],
        "critic": [
            "STATUS: REVISE\nSCORE: 0.2\nFEEDBACK: Flawed.",
            "STATUS: REVISE\nSCORE: 0.3\nFEEDBACK: Still flawed.",
        ],
    }
    link = RecursiveLink(
        codec=LowRankCodec(rank=16),
        hidden_size=64,
        mock_responses=mock_responses,
    )
    orchestrator = RecursiveMASOrchestrator(link, max_rounds=2)
    result = orchestrator.run(task="Unsolvable problem")

    assert result.is_approved is False
    assert result.rounds_executed == 2


def test_orchestrator_summary_and_serialization(tmp_path: Path):
    mock_responses = {
        "planner": ["Step 1: Decompose"],
        "solver": ["Solved"],
        "critic": ["STATUS: APPROVED\nSCORE: 1.0\nFEEDBACK: Good"],
    }
    link = RecursiveLink(
        codec=LowRankCodec(rank=16),
        hidden_size=64,
        mock_responses=mock_responses,
    )
    orchestrator = RecursiveMASOrchestrator(link, max_rounds=1)
    result = orchestrator.run(task="Summary test")

    summary_text = result.format_summary()
    assert "RECURSIVEMAS MULTI-AGENT EXECUTION SUMMARY" in summary_text
    assert "APPROVED" in summary_text
    assert "WIRE BANDWIDTH & LATENCY METRICS" in summary_text

    as_dict = result.as_dict()
    assert as_dict["is_approved"] is True
    assert as_dict["compression_ratio"] > 1.0

    # Test JSON dump
    out_file = tmp_path / "mas_run.json"
    with open(out_file, "w") as f:
        json.dump(as_dict, f)
    assert out_file.exists()


# ==============================================================================
# Live Tiny Sharded Chain Execution Test
# ==============================================================================


def test_orchestrator_with_served_chain(served_agent_chain):
    """Verifies live gRPC sharded chain execution with LowRankCodec and RecursiveMAS."""
    with ChainRunner(served_agent_chain) as runner:
        link = RecursiveLink(
            runner=runner,
            codec=get_codec("lowrank", rank=16),
            hidden_size=64,
        )
        orchestrator = RecursiveMASOrchestrator(link, max_rounds=1, max_new_tokens=4)
        result = orchestrator.run(task="Offline sharded agent execution")

        assert result.rounds_executed == 1
        assert len(result.plan.steps) >= 1
        assert result.wire_bytes_transferred > 0
        assert result.compression_ratio > 1.0


# ==============================================================================
# CLI Argument Parser and Offline Mock Execution Tests
# ==============================================================================


def test_cli_argument_parsing():
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--task",
            "Design P2P mesh",
            "--codec",
            "lowrank",
            "--codec-rank",
            "64",
            "--shards",
            "4",
            "--max-rounds",
            "2",
        ]
    )
    assert args.task == "Design P2P mesh"
    assert args.codec == "lowrank"
    assert args.codec_rank == 64
    assert args.shards == 4
    assert args.max_rounds == 2


def test_cli_offline_execution(tmp_path: Path):
    out_file = tmp_path / "cli_run.json"
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--task",
            "CLI test task",
            "--mock",
            "--codec",
            "lowrank",
            "--codec-rank",
            "16",
            "--max-rounds",
            "1",
            "--out",
            str(out_file),
        ]
    )
    result = run_multiagent(args)

    assert result is not None
    assert out_file.exists()
    with open(out_file) as f:
        data = json.load(f)
    assert data["task"] == "CLI test task"
    assert data["compression_ratio"] > 1.0
