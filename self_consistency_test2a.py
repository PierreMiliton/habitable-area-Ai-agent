from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from models import FinalResult
from self_consistency_check import (
    build_comparison_agent,
    compare_results,
    run_agent,
    save_execution,
)


REFORMULATED_PROMPT_PATH = (
    Path(__file__).with_name("System_Prompt")
    / "system_prompt_reformulated.txt"
)


def run_test2a(
    primary_agent: Agent,
    primary_result: FinalResult,
) -> dict[str, Any]:
    system_prompt = REFORMULATED_PROMPT_PATH.read_text(encoding="utf-8")
    comparison_agent = build_comparison_agent(
        primary_agent.model,
        system_prompt,
    )
    control_result, control_shapes = run_agent(comparison_agent)
    save_execution("test2a", control_result, control_shapes)
    return compare_results(primary_result, control_result, "test2a")
