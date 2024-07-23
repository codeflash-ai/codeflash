import os
import sys
from typing import Any, Dict

import pandas as pd
from sqlalchemy import create_engine


def execute_query(query: str, trace_id: str):
    database_uri = os.environ.get("DATABASE_URL")
    engine = create_engine(database_uri)
    with engine.connect() as connection:
        result = pd.read_sql_query(query, connection, params=[(trace_id,)])
    return result.iloc[0].to_dict() if not result.empty else None


def extract_json_values(json_column: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in json_column.items()}


def write_to_file(filename: str, content: str) -> None:
    with open(filename, "w") as file:
        if isinstance(content, list):
            content = "\n".join(content)
        file.write(str(content))


def main(trace_id: str) -> None:
    query = "SELECT original_code, optimizations_post, speedup_ratio, generated_test, instrumented_generated_test, explanations_post FROM optimization_features WHERE trace_id = %s"
    result = execute_query(query, trace_id)

    if not result:
        print(f"No data found for trace_id: {trace_id}")
        return

    # Extract data from the result
    original_code = result["original_code"]
    optimizations_post = result["optimizations_post"]
    speedup_ratio = result["speedup_ratio"]
    generated_test = result["generated_test"]
    explanations_post = result["explanations_post"]
    instrumented_tests = result["instrumented_generated_test"]

    # Write original code to file
    write_to_file("original_code.py", original_code)

    # Write each optimization candidate to its own file
    for idx, (opt_id, optimization) in enumerate(
        extract_json_values(optimizations_post).items(),
        start=1,
    ):
        filename = f"optimization_candidate_{idx}.py"
        explanation = explanations_post.get(opt_id, "")
        speedup = speedup_ratio.get(opt_id)
        speedup_comment = f"Speedup: {speedup}" if speedup is not None else "No speedup"
        content_with_comment = f'"""{explanation}\n\n{speedup_comment}"""\n\n{optimization}'
        write_to_file(filename, content_with_comment)

    # Find and write the best optimization candidate to its own file
    if speedup_ratio is not None:
        valid_speedup_values = [v for v in speedup_ratio.values() if v is not None]
        best_speedup = max(valid_speedup_values, default=None) if valid_speedup_values else None
        if best_speedup is not None:
            best_optimization_id = next(
                (id for id, speedup in speedup_ratio.items() if speedup == best_speedup),
                None,
            )
            best_optimization = optimizations_post.get(best_optimization_id)
            best_explanation = explanations_post.get(best_optimization_id, "")
            best_speedup_comment = f"Speedup: {best_speedup}"
            best_content_with_comment = (
                f'"""{best_explanation}\n\n{best_speedup_comment}"""\n\n{best_optimization}'
            )
            if best_optimization and best_explanation is not None:
                write_to_file(
                    "best_optimization_candidate.py",
                    best_content_with_comment,
                )
    else:
        print("No speedup ratio found")

    # Write generated tests to file
    write_to_file("generated_tests.py", generated_test)
    write_to_file("instrumented_generated_tests.py", instrumented_tests)


"""
Run the script using the .env file that contains the DATABASE_URL for the optimization_features table
Pass in the trace_id as a command line argument and the script will generate the following files for that trace id:
- original_code.py
- optimization_candidate_1.py, optimization_candidate_2.py, ...
- best_optimization_candidate.py
- generated_tests.py
"""
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_optimization_features.py <trace_id>")
        sys.exit(1)
    trace_id_input = sys.argv[1]
    main(trace_id_input)
