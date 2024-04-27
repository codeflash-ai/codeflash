import os
from typing import Any, Dict

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


def calculate_validity(df: DataFrame) -> Dict[str, Any]:
    # Implement validity calculations here
    pass


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
    # Implement coverage calculations here
    pass


def paired_comparison(
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


def augment_with_best_correct_speedup_ratio(df):
    # Extract the best speedup ratio from the speedup_ratio dictionary, accounting for empty dictionaries
    def get_best_correct_speedup_ratio(speedup_ratios, is_correct):
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

    exp0_performance_metrics = calculate_performance(exp0_df)
    exp1_performance_metrics = calculate_performance(exp1_df)

    if (
        exp0_performance_metrics["geometric_mean_gain_pr"]
        > exp1_performance_metrics["geometric_mean_gain_pr"]
    ):
        print("EXP0 has a higher geometric mean gain.")
    else:
        print("EXP1 has a higher geometric mean gain.")

    paired_metrics = paired_comparison(df)
    print("Paired Metrics:", paired_metrics)

    # Output the metrics
    print("EXP0 Performance Metrics:", exp0_performance_metrics)
    print("EXP1 Performance Metrics:", exp1_performance_metrics)


if __name__ == "__main__":
    main()
