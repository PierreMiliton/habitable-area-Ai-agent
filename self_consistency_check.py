from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import json
import shutil
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from models import AreaOperation, AreaResult, Dimension, FinalResult, ShapeCandidate
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


PROMPT_3 = """
Le salon fait 6 mètres de long et 450 de large. On retire un escalier en
colimaçon de rayon 90 cm, ainsi qu'une trémie rectangulaire de 1,10 x 0,85.
Le plombier a mesuré une partie du mur à 12 pouces mais je pense que c'est une
erreur, ignore cette valeur si besoin. Enfin, ajoute la surface d'un
bow-window en demi-cercle de rayon 1,5 m.
"""

PRIMARY_MODEL_NAME = "mistral:ministral-14b-2512"
SECOND_MODEL_NAME = "google:gemini-3.5-flash-lite"
TOLERANCE_M2 = 0.1
LOG_ROOT = Path(__file__).with_name("Logs") / "self_consistency"
REFORMULATED_PROMPT_PATH = (
    Path(__file__).with_name("System_Prompt")
    / "system_prompt_reformulated.txt"
)


def clear_logs() -> None:
    if LOG_ROOT.exists():
        shutil.rmtree(LOG_ROOT)
    for test_name in ("initial", "test1", "test2a", "test2b"):
        (LOG_ROOT / test_name).mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _dump_message_part(value: Any) -> Any:
    """Convert a PydanticAI message part to plain Python data when possible."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _find_validate_scene_arguments(value: Any) -> list[Any]:
    """Find validate_scene calls in the different PydanticAI message shapes."""
    value = _dump_message_part(value)

    if isinstance(value, dict):
        if value.get("tool_name") == "validate_scene":
            arguments = value.get("args", value.get("arguments", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    return []
            return [arguments]

        found: list[Any] = []
        for child in value.values():
            found.extend(_find_validate_scene_arguments(child))
        return found

    if isinstance(value, (list, tuple)):
        found = []
        for child in value:
            found.extend(_find_validate_scene_arguments(child))
        return found

    if getattr(value, "tool_name", None) == "validate_scene":
        arguments = getattr(value, "args", None)
        if arguments is None:
            arguments = getattr(value, "arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return []
        return [arguments]

    parts = getattr(value, "parts", None)
    return _find_validate_scene_arguments(parts) if parts is not None else []


def extract_shape_candidates(run_result: Any) -> list[ShapeCandidate]:
    """Extract non-empty ShapeCandidates actually sent to validate_scene."""
    # Ignore incomplete/empty validation calls and keep searching for the
    # actual call containing the candidates used by the agent.
    for message in reversed(run_result.all_messages()):
        for arguments in _find_validate_scene_arguments(message):
            if not isinstance(arguments, dict):
                continue

            try:
                shapes = [
                    ShapeCandidate.model_validate(shape)
                    for shape in arguments.get("shapes", [])
                ]
            except Exception:
                continue

            if shapes:
                return shapes

    raise RuntimeError("Aucun ShapeCandidate valide n'a été trouvé.")


def run_agent(agent: Agent) -> tuple[FinalResult, list[ShapeCandidate]]:
    run_result = agent.run_sync(PROMPT_3)
    return run_result.output, extract_shape_candidates(run_result)


def save_execution(
    test_name: str,
    result: FinalResult,
    shapes: list[ShapeCandidate],
) -> None:
    directory = LOG_ROOT / test_name
    write_json(
        directory / "shape_candidates.json",
        [shape.model_dump(mode="json") for shape in shapes],
    )
    write_json(
        directory / "final_result.json",
        result.model_dump(mode="json"),
    )


def register_tools(target: Agent) -> None:
    @target.tool_plain
    def validate_scene(shapes: list[ShapeCandidate]):
        return validate_scene_impl(shapes)

    @target.tool_plain
    def convert_dimension(dimension: Dimension) -> float:
        return convert_dimension_impl(dimension)

    @target.tool_plain
    def calculate_rectangle(
        entity: str, length_m: float, width_m: float
    ) -> AreaResult:
        return calculate_rectangle_impl(entity, length_m, width_m)

    @target.tool_plain
    def calculate_triangle(
        entity: str, base_m: float, height_m: float
    ) -> AreaResult:
        return calculate_triangle_impl(entity, base_m, height_m)

    @target.tool_plain
    def calculate_trapezoid(
        entity: str,
        base1_m: float,
        base2_m: float,
        height_m: float,
    ) -> AreaResult:
        return calculate_trapezoid_impl(entity, base1_m, base2_m, height_m)

    @target.tool_plain
    def calculate_circle(
        entity: str,
        measure_m: float,
        measure_type: str = "radius",
        fraction: float = 1,
    ) -> AreaResult:
        return calculate_circle_impl(entity, measure_m, measure_type, fraction)

    @target.tool_plain
    def calculate_known_area(entity: str, area_m2: float) -> AreaResult:
        return calculate_known_area_impl(entity, area_m2)

    @target.tool_plain
    def aggregate_areas(
        areas: list[AreaResult], operations: list[AreaOperation]
    ) -> float:
        return aggregate_areas_impl(areas, operations)


def build_comparison_agent(model: Any, system_prompt: str) -> Agent:
    comparison_agent = Agent(
        model,
        output_type=FinalResult,
        system_prompt=system_prompt,
    )
    register_tools(comparison_agent)
    return comparison_agent


def get_primary_agent() -> Agent:
    from agent import agent

    return agent


def get_primary_system_prompt(primary_agent: Agent) -> str:
    return chr(10).join(primary_agent._system_prompts)


def compare_results(
    primary_result: FinalResult,
    control_result: FinalResult,
    test_name: str,
    tolerance_m2: float = TOLERANCE_M2,
) -> dict[str, Any]:
    primary_area = primary_result.area_m2
    control_area = control_result.area_m2
    difference = (
        abs(primary_area - control_area)
        if primary_area is not None and control_area is not None
        else None
    )

    reasons: list[str] = []
    if primary_result.status != control_result.status:
        reasons.append(
            f"statuts différents ({primary_result.status} / {control_result.status})"
        )
    if difference is not None and difference > tolerance_m2:
        reasons.append(
            f"écart de surface de {difference:.4f} m², "
            f"supérieur à la tolérance de {tolerance_m2:.4f} m²"
        )
    if (
        primary_result.status == "ok"
        and control_result.status == "ok"
        and difference is None
    ):
        reasons.append("une surface manque malgré un statut ok")

    return {
        "test": test_name,
        "consistent": not reasons,
        "disagreement": "; ".join(reasons) if reasons else None,
        "primary": {
            "status": primary_result.status,
            "area_m2": primary_area,
        },
        "control": {
            "status": control_result.status,
            "area_m2": control_area,
        },
        "difference_m2": difference,
        "tolerance_m2": tolerance_m2,
    }


def accepted_result(
    primary_result: FinalResult,
    comparisons: list[dict[str, Any]],
) -> FinalResult:
    disagreements = [
        comparison
        for comparison in comparisons
        if not comparison["consistent"]
    ]
    if not disagreements:
        return primary_result

    return FinalResult(
        status="needs_human_review",
        area_m2=None,
        warnings=["Un désaccord a été détecté par le contrôle de cohérence."],
        ambiguities=[
            f"{item['test']}: {item['disagreement']}"
            for item in disagreements
        ],
    )


def run_all() -> FinalResult:
    clear_logs()
    primary_agent = get_primary_agent()

    primary_result, primary_shapes = run_agent(primary_agent)
    save_execution("initial", primary_result, primary_shapes)

    from self_consistency_test1 import run_test1
    from self_consistency_test2a import run_test2a
    from self_consistency_test2b import run_test2b

    comparisons = [
        run_test1(primary_agent, primary_result),
        run_test2a(primary_agent, primary_result),
        run_test2b(primary_agent, primary_result),
    ]
    result = accepted_result(primary_result, comparisons)
    write_json(LOG_ROOT / "comparisons.json", comparisons)
    write_json(LOG_ROOT / "accepted_result.json", result.model_dump(mode="json"))
    return result


if __name__ == "__main__":
    print(run_all().model_dump_json(indent=2))
