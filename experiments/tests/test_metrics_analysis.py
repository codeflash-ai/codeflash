import pickle

import pytest
from pandas import DataFrame

from experiments.metrics_analysis import calculate_validity
from experiments.tests.data.mock_dataframe import mock_dataframe
from experiments.tests.data.pie4perf_dataframe_dict import pie4perf_sample_dataframe_dict


def pickled_dataframe():
    with open("experiments/tests/sample_dataframe.pickle", "rb") as f:
        df = pickle.load(f)
    return df


@pytest.fixture
def pie4perf_sample_dataframe():
    return DataFrame.from_dict(pie4perf_sample_dataframe_dict)


@pytest.fixture
def sample_dataframe():
    return DataFrame(mock_dataframe)


def test_calculate_validity_no_successful_runs(pie4perf_sample_dataframe):
    df = pie4perf_sample_dataframe
    df["original_runtime"] = [None, None, None, None, None, None]  # No successful runs
    validity_metrics = calculate_validity(df)
    assert validity_metrics["percent_valid_pr"] == 0
    assert validity_metrics["percent_valid_candidates"] == 0


def test_calculate_validity_some_successful_runs(pie4perf_sample_dataframe):
    df = pie4perf_sample_dataframe
    validity_metrics = calculate_validity(df)
    # Assuming half the runs are successful and above threshold, the expected value should be 50.0
    assert validity_metrics["percent_valid_pr"] == 50.0
    assert validity_metrics["percent_valid_candidates"] == 80.0  # No valid candidates


def test_calculate_validity_all_successful_runs(pie4perf_sample_dataframe):
    df = pie4perf_sample_dataframe
    df["original_runtime"] = [1.0, 2.0, 2.0, 2.0, 3.0, 18.0]  # All successful runs
    df["best_correct_speedup_ratio"] = [0.05, 0.06, 0.08, 0.09, 0.56, 0.17]  # All above threshold
    df["is_correct"] = [
        {"1f39ef86-5eff-4760-a262-43011492906e": True},
        {"aeeeca3b-4ccf-46eb-8dbf-c526c05fca27": True},
        {"dc67527a-9cfe-4bbe-98eb-266bc8e9a27d": True},
        {"e584d267-f68f-4144-820f-110c2719919f": True},
        {"e584d267-f68f-4144-820f-110c2719919f": True},
        {"d34db33f-f68f-4144-820f-110c2719919f": True},
    ]
    validity_metrics = calculate_validity(df)
    assert validity_metrics["percent_valid_pr"] == 100.0
    assert validity_metrics["percent_valid_candidates"] == 100.0  # All valid candidates


def test_calculate_validity_with_valid_candidates(pie4perf_sample_dataframe):
    df = pie4perf_sample_dataframe
    # Assuming some runs are successful and some candidates are valid
    df["original_runtime"] = [1.0, None, 2.0, None, 3.0, None]  # Some successful runs
    df["best_correct_speedup_ratio"] = [0.04, None, 0.06, None, 0.07, None]  # One below and two above the threshold
    df["is_correct"] = [
        {"1f39ef86-5eff-4760-a262-43011492906e": True},
        {},
        {"aeeeca3b-4ccf-46eb-8dbf-c526c05fca27": True},
        {},
        {"e584d267-f68f-4144-820f-110c2719919f": False},
        {},
    ]
    # There are 2 successful runs with valid speedup ratios out of 6 total runs
    expected_percent_valid_pr = (2 / 6) * 100
    # There are 2 valid candidates out of 3 total candidates (one candidate is not valid)
    expected_percent_valid_candidates = (2 / 3) * 100
    validity_metrics = calculate_validity(df)
    assert validity_metrics["percent_valid_pr"] == expected_percent_valid_pr
    assert validity_metrics["percent_valid_candidates"] == expected_percent_valid_candidates


# def test_calculate_performance_no_successful_runs(pie4perf_sample_dataframe):
#     df = pie4perf_sample_dataframe
#     df["original_runtime"] = [nan, nan, nan, nan, nan, nan]  # No successful runs
#     df["best_correct_speedup_ratio"] = [nan, nan, nan, nan, nan, nan]  # No valid candidates
#     performance_metrics = calculate_performance(df)
#     assert performance_metrics["average_percentage_gain_pr"] == 0
#     assert performance_metrics["geometric_mean_gain_pr"] == 0
#     assert performance_metrics["mean_average_percentage_gain_all"] == 0
#     assert performance_metrics["geometric_mean_gain_all"] == 0
#     assert performance_metrics["average_time_saved_pr"] == 0
#     assert performance_metrics["mean_average_time_saved_all"] == 0
#
#
# def test_calculate_performance_some_successful_runs(pie4perf_sample_dataframe):
#     df = pie4perf_sample_dataframe
#     # Assuming half the runs are successful and above threshold, the expected value should be 50.0
#     performance_metrics = calculate_performance(df)
#     assert performance_metrics["average_percentage_gain_pr"] == pytest.approx(0.02112, 0.001)
#     assert performance_metrics["geometric_mean_gain_pr"] == pytest.approx(1.02112, 0.001)
#     assert performance_metrics["mean_average_percentage_gain_all"] == pytest.approx(0.02112, 0.001)
#     assert performance_metrics["geometric_mean_gain_all"] == pytest.approx(1.02112, 0.001)
#     assert performance_metrics["average_time_saved_pr"] == pytest.approx(0.0, 0.001)
#     assert performance_metrics["mean_average_time_saved_all"] == pytest.approx(0.0, 0.001)
#
#
# def test_calculate_performance_all_successful_runs(pie4perf_sample_dataframe):
#     df = pie4perf_sample_dataframe
#     df["original_runtime"] = [1.0, 2.0, 2.0, 2.0, 3.0, 4.0]  # All successful runs
#     df["best_correct_speedup_ratio"] = [
#         0.05,
#         0.06,
#         0.08,
#         0.09,
#         0.56,
#         0.57,
#     ]  # All above threshold
#     df["is_correct"] = [
#         {"1f39ef86-5eff-4760-a262-43011492906e": True},
#         {"aeeeca3b-4ccf-46eb-8dbf-c526c05fca27": True},
#         {"dc67527a-9cfe-4bbe-98eb-266bc8e9a27d": True},
#         {"e584d267-f68f-4144-820f-110c2719919f": True},
#         {"e584d267-f68f-4144-820f-110c2719919f": True},
#         {"d34db33f-f68f-4144-820f-110c2719919f": True},
#     ]
#     performance_metrics = calculate_performance(df)
#     assert performance_metrics["average_percentage_gain_pr"] == pytest.approx(0.235, 0.001)
#     assert performance_metrics["geometric_mean_gain_pr"] == pytest.approx(1.214, 0.001)
#     assert performance_metrics["mean_average_percentage_gain_all"] == pytest.approx(0.235, 0.001)
#     assert performance_metrics["geometric_mean_gain_all"] == pytest.approx(1.214, 0.001)
#     assert performance_metrics["average_time_saved_pr"] == pytest.approx(0.0, 0.001)
#     assert performance_metrics["mean_average_time_saved_all"] == pytest.approx(0.0, 0.001)
#

if __name__ == "__main__":
    pytest.main()
