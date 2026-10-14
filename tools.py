from __future__ import annotations

from geometry import (
    circle_area,
    convert_to_meters,
    rectangle_area,
    trapezoid_area,
    triangle_area,
)
from models import (
    AreaOperation,
    AreaResult,
    Dimension,
    ShapeCandidate,
    ValidationResult,
)


REQUIRED_DIMENSIONS = {
    "rectangle": ("length", "width"),
    "triangle": ("base", "height"),
    "trapezoid": ("height", "base1", "base2"),
}


def validate_scene(shapes: list[ShapeCandidate]) -> ValidationResult:
    """Check that every shape has the dimensions needed by its formula."""
    valid: list[ShapeCandidate] = []
    ambiguities: list[str] = []
    warnings: list[str] = []

    for shape in shapes:
        if shape.operation == "ignore":
            warnings.append(f"{shape.entity}: mesure explicitement ignorée.")
            continue

        if shape.type == "unknown":
            ambiguities.append(
                shape.operation_justification
                or f"{shape.entity}: géométrie ou mesure nécessaire indéterminée."
            )
            continue

        if shape.operation == "uncertain":
            ambiguities.append(
                shape.operation_justification
                or f"{shape.entity}: contribution à la surface habitable indéterminée."
            )
            continue

        if shape.type == "known_area":
            if shape.known_area_m2 is None:
                ambiguities.append(f"{shape.entity}: surface fournie sans valeur.")
            else:
                valid.append(shape)
            continue

        if shape.type == "circle":
            missing = not ("radius" in shape.dimensions or "diameter" in shape.dimensions)
            if missing:
                ambiguities.append(f"{shape.entity}: rayon ou diamètre manquant.")
            elif "radius" in shape.dimensions and "diameter" in shape.dimensions:
                warnings.append(f"{shape.entity}: rayon et diamètre fournis ; seul le rayon sera utilisé.")
        else:
            missing_names = [name for name in REQUIRED_DIMENSIONS[shape.type] if name not in shape.dimensions]
            if missing_names:
                ambiguities.append(f"{shape.entity}: dimensions manquantes : {', '.join(missing_names)}.")

        if not any(message.startswith(f"{shape.entity}:") for message in ambiguities):
            valid.append(shape)

    return ValidationResult(valid_shapes=valid, ambiguities=ambiguities, warnings=warnings)


def convert_dimension(dimension: Dimension) -> float:
    """Convert one dimension to metres; no conversion state is needed."""
    return convert_to_meters(dimension.value, dimension.unit)


def _area(entity: str, value: float) -> AreaResult:
    return AreaResult(entity=entity, area_m2=value)


def calculate_rectangle(entity: str, length_m: float, width_m: float) -> AreaResult:
    return _area(entity, rectangle_area(length_m, width_m))


def calculate_triangle(entity: str, base_m: float, height_m: float) -> AreaResult:
    return _area(entity, triangle_area(base_m, height_m))


def calculate_trapezoid(entity: str, base1_m: float, base2_m: float, height_m: float) -> AreaResult:
    return _area(entity, trapezoid_area(base1_m, base2_m, height_m))


def calculate_circle(
    entity: str, measure_m: float, measure_type: str = "radius", fraction: float = 1
) -> AreaResult:
    if measure_type not in {"radius", "diameter"}:
        raise ValueError("measure_type doit être 'radius' ou 'diameter'.")
    if not 0 < fraction <= 1:
        raise ValueError("fraction doit être comprise entre 0 et 1.")
    return _area(entity, circle_area(measure_m, measure_type, fraction))


def calculate_known_area(entity: str, area_m2: float) -> AreaResult:
    return _area(entity, area_m2)


def aggregate_areas(
    areas: list[AreaResult],
    operations: list[AreaOperation],
) -> float:
    by_entity = {area.entity: area.area_m2 for area in areas}
    missing = [op.entity for op in operations if op.entity not in by_entity]
    if missing:
        raise ValueError(f"Aires non calculées : {', '.join(missing)}")

    total = sum(
        by_entity[op.entity] if op.operation == "add" else -by_entity[op.entity]
        for op in operations
    )
    if total < 0:
        raise ValueError("La surface finale est négative ; vérifier les opérations.")
    return total
