from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02343_0():
    input_content = (
        "5 12\n0 1 4\n0 2 3\n1 1 2\n1 3 4\n1 1 4\n1 3 2\n0 1 3\n1 2 4\n1 3 0\n0 0 4\n1 0 2\n1 3 0"
    )
    expected_output = "0\n0\n1\n1\n1\n0\n1\n1"
    run_pie_test_case("../p02343.py", input_content, expected_output)


def test_problem_p02343_1():
    input_content = (
        "5 12\n0 1 4\n0 2 3\n1 1 2\n1 3 4\n1 1 4\n1 3 2\n0 1 3\n1 2 4\n1 3 0\n0 0 4\n1 0 2\n1 3 0"
    )
    expected_output = "0\n0\n1\n1\n1\n0\n1\n1"
    run_pie_test_case("../p02343.py", input_content, expected_output)
