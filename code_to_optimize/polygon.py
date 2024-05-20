from functools import cached_property
from typing import List, Tuple
import numpy as np
from dataclasses import dataclass

Point = np.array


@dataclass
class Polygon:
    x: np.array
    y: np.array
    _area: float = None

    @cached_property
    def center(self) -> np.array:
        centroid = np.array([self.x, self.y]).mean(axis=1)
        return centroid

    def area(self) -> float:
        if self._area is None:
            self._area = 0.5 * np.abs(
                np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1))
            )
        return self._area


def find_close_polygons(
    polygon_subset: List[Polygon], point: np.array, max_dist: float
) -> List[Polygon]:
    close_polygons = []
    for poly in polygon_subset:
        if np.linalg.norm(poly.center - point) < max_dist:
            close_polygons.append(poly)

    return close_polygons


def select_best_polygon(
    polygon_sets: List[Tuple[Point, List[Polygon]]]
) -> List[Tuple[Point, Polygon]]:
    best_polygons = []
    for point, polygons in polygon_sets:
        best_polygon = polygons[0]

        for poly in polygons:
            if poly.area() < best_polygon.area():
                best_polygon = poly

        best_polygons.append((point, best_polygon))

    return best_polygons


def main(polygons: List[Polygon], points: np.ndarray) -> List[Tuple[Point, Polygon]]:
    max_dist = 10.0
    polygon_sets = []
    for point in points:
        close_polygons = find_close_polygons(polygons, point, max_dist)

        if len(close_polygons) == 0:
            continue

        polygon_sets.append((point, close_polygons))

    best_polygons = select_best_polygon(polygon_sets)

    return best_polygons
