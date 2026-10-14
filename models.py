from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


Unit = Literal["mm", "cm", "m", "in", "ft"]


class Dimension(BaseModel):
    value: float = Field(gt=0)
    unit: Unit


class ShapeCandidate(BaseModel):
    """A shape extracted by the LLM from the inspector's description."""

    entity: str
    type: Literal[
        "rectangle",
        "triangle",
        "trapezoid",
        "circle",
        "known_area",
        "unknown",
    ]
    # "uncertain" means that the contribution to habitable area is unclear.
    # It is distinct from type="unknown", which means the geometry itself
    # cannot be calculated.
    operation: Literal["add", "subtract", "ignore", "uncertain"]
    operation_justification: str = ""
    dimensions: dict[str, Dimension] = Field(default_factory=dict)
    known_area_m2: float | None = Field(default=None, gt=0)
    # 1 = full circle, 0.5 = half circle, 0.25 = quarter circle.
    fraction: float = Field(default=1, gt=0, le=1)


class ValidationResult(BaseModel):
    valid_shapes: list[ShapeCandidate] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AreaResult(BaseModel):
    entity: str
    area_m2: float = Field(ge=0)


class AreaOperation(BaseModel):
    entity: str
    operation: Literal["add", "subtract"]


class FinalResult(BaseModel):
    status: Literal["ok", "needs_human_review"]
    area_m2: float | None = None
    warnings: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def normalize_review_result(self) -> "FinalResult":
        self.warnings = list(dict.fromkeys(item.strip() for item in self.warnings if item.strip()))
        self.ambiguities = list(
            dict.fromkeys(item.strip() for item in self.ambiguities if item.strip())
        )
        if self.status == "needs_human_review":
            self.area_m2 = None
        return self
