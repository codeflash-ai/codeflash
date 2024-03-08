from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03632_0():
    input_content = "0 75 25 100"
    expected_output = "50"
    run_pie_test_case("../p03632.py", input_content, expected_output)


def test_problem_p03632_1():
    input_content = "10 90 20 80"
    expected_output = "60"
    run_pie_test_case("../p03632.py", input_content, expected_output)


def test_problem_p03632_2():
    input_content = "0 75 25 100"
    expected_output = "50"
    run_pie_test_case("../p03632.py", input_content, expected_output)


def test_problem_p03632_3():
    input_content = "0 33 66 99"
    expected_output = "0"
    run_pie_test_case("../p03632.py", input_content, expected_output)
