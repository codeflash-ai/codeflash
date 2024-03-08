from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02295_0():
    input_content = "3\n0 0 2 0 1 1 1 -1\n0 0 1 1 0 1 1 0\n0 0 1 1 1 0 0 1"
    expected_output = (
        "1.0000000000 0.0000000000\n0.5000000000 0.5000000000\n0.5000000000 0.5000000000"
    )
    run_pie_test_case("../p02295.py", input_content, expected_output)


def test_problem_p02295_1():
    input_content = "3\n0 0 2 0 1 1 1 -1\n0 0 1 1 0 1 1 0\n0 0 1 1 1 0 0 1"
    expected_output = (
        "1.0000000000 0.0000000000\n0.5000000000 0.5000000000\n0.5000000000 0.5000000000"
    )
    run_pie_test_case("../p02295.py", input_content, expected_output)
