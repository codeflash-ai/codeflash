from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02290_0():
    input_content = "0 0 2 0\n3\n-1 1\n0 1\n1 1"
    expected_output = (
        "-1.0000000000 0.0000000000\n0.0000000000 0.0000000000\n1.0000000000 0.0000000000"
    )
    run_pie_test_case("../p02290.py", input_content, expected_output)


def test_problem_p02290_1():
    input_content = "0 0 3 4\n1\n2 5"
    expected_output = "3.1200000000 4.1600000000"
    run_pie_test_case("../p02290.py", input_content, expected_output)


def test_problem_p02290_2():
    input_content = "0 0 2 0\n3\n-1 1\n0 1\n1 1"
    expected_output = (
        "-1.0000000000 0.0000000000\n0.0000000000 0.0000000000\n1.0000000000 0.0000000000"
    )
    run_pie_test_case("../p02290.py", input_content, expected_output)
