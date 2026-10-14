import math


UNIT_TO_METER = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
    "in": 0.0254,
    "ft": 0.3048,
}


def convert_to_meters(value: float, unit: str) -> float:
    try:
        return value * UNIT_TO_METER[unit]
    except KeyError as exc:
        raise ValueError(f"Unité inconnue : {unit}") from exc


def rectangle_area(length_m: float, width_m: float) -> float:
    return length_m * width_m


def triangle_area(base_m: float, height_m: float) -> float:
    return base_m * height_m / 2


def trapezoid_area(base1_m: float, base2_m: float, height_m: float) -> float:
    return (base1_m + base2_m) * height_m / 2


def circle_area(measure_m: float, measure_type: str = "radius", fraction: float = 1) -> float:
    """Area of a circle portion. measure_type also handles a diameter directly."""
    radius_m = measure_m if measure_type == "radius" else measure_m / 2
    return math.pi * radius_m**2 * fraction
