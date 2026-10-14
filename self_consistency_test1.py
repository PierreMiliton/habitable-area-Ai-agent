from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

from models import FinalResult
from self_consistency_check import (
    compare_results,
    run_agent,
    save_execution,
)


def run_test1(
    primary_agent: Agent,
    primary_result: FinalResult,
) -> dict[str, Any]:
    rerun_result, rerun_shapes = run_agent(primary_agent)
    save_execution("test1", rerun_result, rerun_shapes)
    return compare_results(primary_result, rerun_result, "test1")
