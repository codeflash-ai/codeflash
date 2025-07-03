from typing import (
    Tuple,
)
from typing_extensions import TypeAlias

Box: TypeAlias = Tuple[float, float, float, float]
Point: TypeAlias = Tuple[float, float]
Points: TypeAlias = Tuple[Point, ...]