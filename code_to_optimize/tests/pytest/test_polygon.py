from code_to_optimize.polygon import (
    Polygon,
    find_close_polygons,
    select_best_polygon,
    main,
)
import numpy as np


def test_simple_selection():
    polygons = [
        Polygon(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1])),
        Polygon(np.array([10, 11, 11, 10]), np.array([10, 10, 11, 11])),
        Polygon(np.array([5, 6, 6, 5]), np.array([5, 5, 6, 6])),
    ]
    point = np.array([1, 1])
    max_dist = 3
    result = find_close_polygons(polygons, point, max_dist)
    assert len(result) == 1
    result[0].center.tolist() == [0.5, 0.5]


def test_no_close_polygons():
    polygons = [
        Polygon(np.array([10, 11, 11, 10]), np.array([10, 10, 11, 11])),
        Polygon(np.array([20, 21, 21, 20]), np.array([20, 20, 21, 21])),
    ]
    point = np.array([1, 1])
    max_dist = 5
    result = find_close_polygons(polygons, point, max_dist)
    assert len(result) == 0


def test_distinct_areas():
    point1 = np.array([0, 0])
    polygons = [
        Polygon(np.array([0, 1, 0]), np.array([0, 0, 1])),  # Area: 0.5
        Polygon(np.array([0, 2, 0]), np.array([0, 0, 2])),  # Area: 2.0
        Polygon(np.array([0, 0.5, 0]), np.array([0, 0, 0.5])),  # Area: 0.125
    ]
    result = select_best_polygon([(point1, polygons)])
    assert result[0][1] == polygons[2]


def test_tied_areas():
    point1 = np.array([1, 1])
    polygons = [
        Polygon(np.array([0, 1, 0]), np.array([0, 0, 1])),
        Polygon(
            np.array([0, 1, 0]), np.array([0, 0, 1])
        ),  # Identical area as the first
        Polygon(np.array([0, 2, 0]), np.array([0, 0, 2])),  # Larger area
    ]
    result = select_best_polygon([(point1, polygons)])
    assert result[0][1] == polygons[0]


def test_single_closest_polygon_per_point():
    # Setup polygons and points
    polygons = [
        Polygon(np.array([0, 1, 0]), np.array([0, 0, 1])),  # Near origin
        Polygon(np.array([100, 101, 100]), np.array([100, 100, 101])),  # Far away
    ]
    points = np.array([[0.5, 0.5], [150, 150]])  # One point near origin, one far

    # Expected to find only the first polygon near the first point
    expected_results = [(np.array([0.5, 0.5]), polygons[0])]
    results = main(polygons, points)
    assert len(results) == 1
    assert results[0][0].tolist() == expected_results[0][0].tolist()
    assert results[0][1] == expected_results[0][1]


def test_multiple_polygons_close_to_points_select_smallest():
    # Setup polygons and points
    polygons = [
        Polygon(np.array([0, 1, 0]), np.array([0, 0, 1])),  # Area small
        Polygon(np.array([0, 2, 0]), np.array([0, 0, 2])),  # Area large
        Polygon(
            np.array([1, 2, 1]), np.array([1, 1, 2])
        ),  # Area small, close to point 2
    ]
    points = np.array([[0.5, 0.5], [1.5, 1.5]])  # Both points near polygons

    # Expected to find the smallest area polygon for both points
    expected_results = [
        (np.array([0.5, 0.5]), polygons[0]),
        (np.array([1.5, 1.5]), polygons[2]),
    ]
    results = main(polygons, points)
    assert len(results) == 2
    assert results[0][0].tolist() == expected_results[0][0].tolist()
    assert results[0][1] == expected_results[0][1]
    assert results[1][0].tolist() == expected_results[1][0].tolist()
