import os
from typing import Any, Dict, Optional

import pandas as pd
from pandas import DataFrame
from scipy.stats import gmean
from sqlalchemy import create_engine


def load_data(database_uri: str = os.environ.get("DATABASE_URL")) -> DataFrame:
    engine = create_engine(database_uri)
    with engine.connect() as connection:
        query = """
            SELECT * FROM optimization_features
            WHERE (trace_id LIKE %s OR trace_id LIKE %s)
            AND created_at BETWEEN %s AND %s
        """
        return pd.read_sql_query(
            query,
            connection,
            params=("%EXP0", "%EXP1", "2024-04-18 05:51:17.050279+00", "2024-04-18 17:30:53.843066+00"),
        )


def process_column_pairs(df: DataFrame, column_name: str) -> DataFrame:
    """Cleans up the given column by filling in missing values from the other column in the EXP0 / EXP1 pair.
    :param df:
    :param column_name:
    :return:
    """
    new_df = df.copy()
    grouped = new_df.groupby(new_df["trace_id"].str[:-4])
    for _, group in grouped:
        if len(group) == 2:
            exp0_row = group[group["trace_id"].str.endswith("EXP0")]
            exp1_row = group[group["trace_id"].str.endswith("EXP1")]
            if pd.isnull(exp0_row.iloc[0][column_name]) and not pd.isnull(exp1_row.iloc[0][column_name]):
                new_df.at[exp0_row.index[0], column_name] = exp1_row.iloc[0][column_name]
            elif pd.isnull(exp1_row.iloc[0][column_name]) and not pd.isnull(exp0_row.iloc[0][column_name]):
                new_df.at[exp1_row.index[0], column_name] = exp0_row.iloc[0][column_name]
    return new_df


def calculate_validity(df: DataFrame, perf_threshold: float = 0.05) -> Dict[str, Any]:
    # Calculate the percentage of valid PRs given that the original function run succeeded
    successful_runs = df[(~df["original_runtime"].isna())]
    successful_runs_above_thres = successful_runs[
        successful_runs["best_correct_speedup_ratio"] >= perf_threshold
    ]
    valid_prs = len(successful_runs_above_thres)
    percent_valid_pr = (
        valid_prs / len(successful_runs_above_thres) * 100 if len(successful_runs_above_thres) > 0 else 0
    )

    # Calculate the percentage of valid candidates generated given that original function run succeeded
    valid_candidates = successful_runs[successful_runs["is_correct"].apply(lambda x: any(x.values()))]
    percent_valid_candidates = (
        len(valid_candidates) / len(successful_runs) * 100 if len(successful_runs) > 0 else 0
    )

    return {
        "percent_valid_pr": percent_valid_pr,
        "percent_valid_candidates": percent_valid_candidates,
    }


def calculate_performance(df: DataFrame, perf_threshold: float = 0.05) -> Dict[str, Any]:
    # Filter out the rows where a valid candidate above the perf threshold was found
    valid_candidates_above_thres = df[(df["best_correct_speedup_ratio"] >= perf_threshold)]

    # (1) Calculate the average speedup ratio of the PR
    pr_gain = valid_candidates_above_thres["best_correct_speedup_ratio"].mean()

    # (1a) Calculate the geometric mean of the PR speedup ratio with ref=1x
    pr_geom_mean = gmean(valid_candidates_above_thres["best_correct_speedup_ratio"] + 1)

    # (2) Calculate the mean average speedup ratio of all the valid candidates
    mean_average_percentage_gain_all = df["best_correct_speedup_ratio"].mean()

    # (2a) Calculate the geometric mean of all candidates speedup ratio with ref=1x
    all_candidates_geom_mean = gmean(df["best_correct_speedup_ratio"].dropna() + 1)

    def calculate_time_saved_for_row(row: pd.Series):
        if row["optimized_runtime"] is not None and row["is_correct"] is not None:
            correct_runtimes = [
                runtime
                for opt_id, runtime in row["optimized_runtime"].items()
                if row["is_correct"].get(opt_id)
            ]
        else:
            correct_runtimes = []
        if correct_runtimes:
            return row["original_runtime"] - min(correct_runtimes)
        return None

    # (3) The average time saved in a PR given that a valid candidate was found above the perf threshold.
    pr_time_saved = (
        valid_candidates_above_thres.apply(
            lambda row: calculate_time_saved_for_row(row),
            axis=1,
        )
        .dropna()
        .mean()
    )

    # (4) Calculate the mean average time saved for all the valid candidates
    all_candidates_time_saved = (
        df.apply(
            lambda row: calculate_time_saved_for_row(row),
            axis=1,
        )
        .dropna()
        .mean()
    )

    return {
        "average_percentage_gain_pr": pr_gain,
        "geometric_mean_gain_pr": pr_geom_mean,
        "mean_average_percentage_gain_all": mean_average_percentage_gain_all,
        "geometric_mean_gain_all": all_candidates_geom_mean,
        "average_time_saved_pr": pr_time_saved,
        "mean_average_time_saved_all": all_candidates_time_saved,
    }


def calculate_coverage(df: DataFrame) -> Dict[str, Any]:
    successful_runs = df[~df["original_runtime"].isna()]

    def calculate_percent_optimization_successful_runs(opt_runs: Dict[str, Optional[float]]) -> float:
        if opt_runs is None:
            return 0.0
        total_runs = len(opt_runs)
        successful_optimization_runs = sum(1 for runtime in opt_runs.values() if runtime is not None)
        return successful_optimization_runs / total_runs * 100 if total_runs > 0 else 0.0

    df["percent_successful_optimization_runs"] = df["optimized_runtime"].apply(
        calculate_percent_optimization_successful_runs,
    )

    total_optimizations = sum(len(runs) for runs in successful_runs["optimized_runtime"] if runs is not None)
    successful_optimizations = sum(
        len([runtime for runtime in runs.values() if runtime is not None])
        for runs in successful_runs["optimized_runtime"]
        if runs is not None
    )

    average_percent_successful_optimization_runs = df["percent_successful_optimization_runs"].mean()

    percent_successful_optimizations = (
        successful_optimizations / total_optimizations * 100 if total_optimizations > 0 else 0
    )

    percent_successful_original_runs = len(successful_runs) / len(df) * 100

    return {
        "average_percent_successful_optimization_runs": average_percent_successful_optimization_runs,
        "percent_successful_optimizations": percent_successful_optimizations,
        "percent_successful_original_runs": percent_successful_original_runs,
    }


def paired_comparison_coverage(
    df: DataFrame,
    model_a_suffix: str = "EXP0",
    model_b_suffix: str = "EXP1",
) -> Dict[str, Any]:
    paired_coverage_results = {
        "model_a_more_successful": 0,
        "equal_successful": 0,
        "model_b_more_successful": 0,
    }
    grouped = df.groupby(df["trace_id"].str[:-4])
    for _, group in grouped:
        if len(group) == 2:
            model_a_row = group[group["trace_id"].str.endswith(model_a_suffix)]
            model_b_row = group[group["trace_id"].str.endswith(model_b_suffix)]
            if model_a_row["optimized_runtime"].values[0] is None:
                model_a_success_count = 0
            else:
                model_a_success_count = sum(
                    1
                    for runtime in model_a_row["optimized_runtime"].values[0].values()
                    if runtime is not None
                )
            if model_b_row["optimized_runtime"].values[0] is None:
                model_b_success_count = 0
            else:
                model_b_success_count = sum(
                    1
                    for runtime in model_b_row["optimized_runtime"].values[0].values()
                    if runtime is not None
                )

            if model_a_success_count > model_b_success_count:
                paired_coverage_results["model_a_more_successful"] += 1
            elif model_a_success_count < model_b_success_count:
                paired_coverage_results["model_b_more_successful"] += 1
            else:
                paired_coverage_results["equal_successful"] += 1
    return paired_coverage_results

    # Implement coverage calculations here


def paired_comparison_validity(
    df: DataFrame,
    model_a_suffix: str = "EXP0",
    model_b_suffix: str = "EXP1",
) -> Dict[str, Any]:
    # Paired - Calculate the percentage of runs where model A generated more, equal, or less valid candidates than model B
    paired_validity_results = {
        "model_a_more_valid": 0,
        "equal_valid": 0,
        "model_b_more_valid": 0,
    }
    grouped = df.groupby(df["trace_id"].str[:-4])
    for _, group in grouped:
        if len(group) == 2:
            model_a_row = group[group["trace_id"].str.endswith(model_a_suffix)]
            model_b_row = group[group["trace_id"].str.endswith(model_b_suffix)]
            if model_a_row["is_correct"].values[0] is None:
                model_a_valid_count = 0
            else:
                model_a_valid_count = sum(model_a_row["is_correct"].values[0].values())
            if model_b_row["is_correct"].values[0] is None:
                model_b_valid_count = 0
            else:
                model_b_valid_count = sum(model_b_row["is_correct"].values[0].values())

            if model_a_valid_count > model_b_valid_count:
                paired_validity_results["model_a_more_valid"] += 1
            elif model_a_valid_count < model_b_valid_count:
                paired_validity_results["model_b_more_valid"] += 1
            else:
                paired_validity_results["equal_valid"] += 1
    return paired_validity_results


def paired_comparison_performance(
    df: DataFrame,
    model_a_suffix: str = "EXP0",
    model_b_suffix: str = "EXP1",
) -> Dict[str, Any]:
    paired_results = {
        "model_a_better": 0,
        "equal": 0,
        "model_b_better": 0,
    }

    # Group by the trace_id without the suffix
    grouped = df.groupby(df["trace_id"].str[:-4])
    for _, group in grouped:
        if len(group) == 2:
            model_a_row = group[group["trace_id"].str.endswith(model_a_suffix)]
            model_b_row = group[group["trace_id"].str.endswith(model_b_suffix)]
            model_a_speedup = model_a_row["best_correct_speedup_ratio"].values[0]
            model_b_speedup = model_b_row["best_correct_speedup_ratio"].values[0]

            if pd.isna(model_a_speedup) and not pd.isna(model_b_speedup):
                paired_results["model_b_better"] += 1
            elif not pd.isna(model_a_speedup) and pd.isna(model_b_speedup):
                paired_results["model_a_better"] += 1
            elif pd.isna(model_a_speedup) and pd.isna(model_b_speedup):
                # If both are NaN, do nothing
                pass
            elif model_a_speedup > model_b_speedup:
                paired_results["model_a_better"] += 1
            elif model_a_speedup < model_b_speedup:
                paired_results["model_b_better"] += 1
            else:
                paired_results["equal"] += 1

    return paired_results


def augment_with_best_correct_speedup_ratio(df: DataFrame) -> DataFrame:
    # Extract the best speedup ratio from the speedup_ratio dictionary, accounting for empty dictionaries
    def get_best_correct_speedup_ratio(
        speedup_ratios: Dict[str, float],
        is_correct: Dict[str, bool],
    ) -> Optional[float]:
        correct_speedup_ratios = (
            {uuid: ratio for uuid, ratio in speedup_ratios.items() if is_correct.get(uuid)}
            if speedup_ratios is not None
            else {}
        )

        if correct_speedup_ratios:
            return max(correct_speedup_ratios.values())
        return None

    df["best_correct_speedup_ratio"] = df.apply(
        lambda row: get_best_correct_speedup_ratio(
            row["speedup_ratio"],
            row["is_correct"],
        )
        if row["speedup_ratio"] is not None
        else None,
        axis=1,
    )

    return df


def main() -> None:
    df = load_data()
    df = process_column_pairs(df, "metadata")
    df = process_column_pairs(df, "test_framework")
    df = process_column_pairs(df, "generated_test")
    df = augment_with_best_correct_speedup_ratio(df)
    exp0_df = df[df["trace_id"].str.endswith("EXP0")]
    exp1_df = df[df["trace_id"].str.endswith("EXP1")]

    # Calculate metrics for each experiment
    exp0_performance_metrics = calculate_performance(exp0_df)
    exp1_performance_metrics = calculate_performance(exp1_df)
    exp0_validity_metrics = calculate_validity(exp0_df)
    exp1_validity_metrics = calculate_validity(exp1_df)
    exp0_coverage_metrics = calculate_coverage(exp0_df)
    exp1_coverage_metrics = calculate_coverage(exp1_df)

    paired_performance_metrics = paired_comparison_performance(df)
    paired_validity_metrics = paired_comparison_validity(df)
    paired_coverage_metrics = paired_comparison_coverage(df)

    # Combine metrics into a DataFrame
    metrics_df = pd.DataFrame(
        {
            "EXP0": {
                **exp0_performance_metrics,
                **exp0_validity_metrics,
                **exp0_coverage_metrics,
            },
            "EXP1": {
                **exp1_performance_metrics,
                **exp1_validity_metrics,
                **exp1_coverage_metrics,
            },
        },
    ).T  # Transpose to have experiments as rows and metrics as columns

    # Output the combined metrics DataFrame
    print(metrics_df)


if __name__ == "__main__":
    main()
