from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03040_0():
    input_content = "4\n1 4 2\n2\n1 1 -8\n2"
    expected_output = "4 2\n1 -3"
    run_pie_test_case("../p03040.py", input_content, expected_output)


def test_problem_p03040_1():
    input_content = (
        "4\n1 -1000000000 1000000000\n1 -1000000000 1000000000\n1 -1000000000 1000000000\n2"
    )
    expected_output = "-1000000000 3000000000"
    run_pie_test_case("../p03040.py", input_content, expected_output)


def test_problem_p03040_2():
    input_content = "4\n1 4 2\n2\n1 1 -8\n2"
    expected_output = "4 2\n1 -3"
    run_pie_test_case("../p03040.py", input_content, expected_output)
