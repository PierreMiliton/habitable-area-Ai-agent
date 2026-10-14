from __future__ import annotations

import json
from pathlib import Path

from pydantic_ai import Agent, ModelRetry

from models import (
    AreaOperation,
    AreaResult,
    Dimension,
    FinalResult,
    ShapeCandidate,
)
from tools import (
    aggregate_areas as aggregate_areas_impl,
    calculate_circle as calculate_circle_impl,
    calculate_known_area as calculate_known_area_impl,
    calculate_rectangle as calculate_rectangle_impl,
    calculate_trapezoid as calculate_trapezoid_impl,
    calculate_triangle as calculate_triangle_impl,
    convert_dimension as convert_dimension_impl,
    validate_scene as validate_scene_impl,
)


BASE_DIR = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = BASE_DIR / "System_Prompt" / "system_prompt.txt"
LOG_DIR = BASE_DIR / "Logs" / "log_agent"
SHAPE_CANDIDATES_LOG = LOG_DIR / "shape_candidates.json"
FINAL_RESULT_LOG = LOG_DIR / "final_result.json"

SYSTEM_PROMPT = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

_last_shape_candidates: list[ShapeCandidate] = []


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def reset_agent_logs() -> None:
    """Start a fresh log for the three prompts run by main.py."""
    global _last_shape_candidates
    _last_shape_candidates = []
    _write_json(SHAPE_CANDIDATES_LOG, {})
    _write_json(FINAL_RESULT_LOG, {})


def save_agent_result(case_name: str, result: FinalResult) -> None:
    """Save the latest ShapeCandidates and FinalResult for one case."""
    shape_log = (
        json.loads(SHAPE_CANDIDATES_LOG.read_text(encoding="utf-8"))
        if SHAPE_CANDIDATES_LOG.exists()
        else {}
    )
    result_log = (
        json.loads(FINAL_RESULT_LOG.read_text(encoding="utf-8"))
        if FINAL_RESULT_LOG.exists()
        else {}
    )

    shape_log[case_name] = [
        shape.model_dump(mode="json")
        for shape in _last_shape_candidates
    ]
    result_log[case_name] = result.model_dump(mode="json")

    _write_json(SHAPE_CANDIDATES_LOG, shape_log)
    _write_json(FINAL_RESULT_LOG, result_log)

    _last_shape_candidates.clear()


## Agent definition

agent = Agent(
    "mistral:ministral-14b-2512",
    output_type=FinalResult,
    system_prompt=SYSTEM_PROMPT,
)


@agent.tool_plain
def validate_scene(shapes: list[ShapeCandidate]):
    global _last_shape_candidates
    for shape in shapes:
        existing_index = next(
            (
                index
                for index, previous in enumerate(_last_shape_candidates)
                if previous.entity == shape.entity
            ),
            None,
        )
        if existing_index is None:
            _last_shape_candidates.append(shape)
        else:
            _last_shape_candidates[existing_index] = shape
    return validate_scene_impl(shapes)


@agent.tool_plain
def convert_dimension(dimension: Dimension) -> float:
    return convert_dimension_impl(dimension)


@agent.tool_plain
def calculate_rectangle(entity: str, length_m: float, width_m: float) -> AreaResult:
    return calculate_rectangle_impl(entity, length_m, width_m)


@agent.tool_plain
def calculate_triangle(entity: str, base_m: float, height_m: float) -> AreaResult:
    return calculate_triangle_impl(entity, base_m, height_m)


@agent.tool_plain
def calculate_trapezoid(
    entity: str, base1_m: float, base2_m: float, height_m: float
) -> AreaResult:
    return calculate_trapezoid_impl(entity, base1_m, base2_m, height_m)


@agent.tool_plain
def calculate_circle(
    entity: str,
    measure_m: float,
    measure_type: str = "radius",
    fraction: float = 1,
) -> AreaResult:
    return calculate_circle_impl(entity, measure_m, measure_type, fraction)


@agent.tool_plain
def calculate_known_area(entity: str, area_m2: float) -> AreaResult:
    return calculate_known_area_impl(entity, area_m2)


@agent.tool_plain
def aggregate_areas(
    areas: list[AreaResult], operations: list[AreaOperation]
) -> float:
    try:
        return aggregate_areas_impl(areas, operations)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc
