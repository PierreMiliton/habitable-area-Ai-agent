from __future__ import annotations

from typing import Any

from pydantic_ai import Agent

from models import FinalResult
from self_consistency_check import (
    SECOND_MODEL_NAME,
    build_comparison_agent,
    compare_results,
    get_primary_system_prompt,
    run_agent,
    save_execution,
)


def run_test2b(
    primary_agent: Agent,
    primary_result: FinalResult,
) -> dict[str, Any]:
    comparison_agent = build_comparison_agent(
        SECOND_MODEL_NAME,
        get_primary_system_prompt(primary_agent),
    )
    control_result, control_shapes = run_agent(comparison_agent)
    save_execution("test2b", control_result, control_shapes)
    return compare_results(primary_result, control_result, "test2b")
