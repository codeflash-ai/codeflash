from code_to_optimize.gradient import gradient
import numpy as np


def test_simple_case():
    # Test case with simple values
    n_features = 2
    n_samples = 2
    y = np.array([1, -1])
    X = np.array([[1, 2], [3, 4]])
    w = np.array([0.5, -0.5])
    b = 0.1
    subgrad = np.zeros(n_features)
    lambda1 = 0.1
    lambda2 = 0.05

    # Expected result calculated manually or by a reliable source
    expected_subgrad = np.array([2.15, 1.85])

    # Perform the function call
    result = gradient(n_features, n_samples, y, X, w, b, subgrad, lambda1, lambda2)

    # Assert to check if expected result is the actual result
    np.testing.assert_array_almost_equal(result, expected_subgrad, decimal=5)


def test_edge_case():
    n_features = 2
    n_samples = 1
    y = np.array([1])
    X = np.array([[10, -10]])
    w = np.array([1, -1])
    b = -100
    subgrad = np.zeros(n_features)
    lambda1 = 0.1
    lambda2 = 0.05

    # All examples correctly classified with a large margin
    expected_subgrad = np.array([-9.8, 9.8])

    # Perform the function call
    result = gradient(n_features, n_samples, y, X, w, b, subgrad, lambda1, lambda2)

    # Assert
    np.testing.assert_array_almost_equal(result, expected_subgrad, decimal=5)
