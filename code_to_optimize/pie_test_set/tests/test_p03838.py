from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03838_0():
    input_content = "10 20"
    expected_output = "10"
    run_pie_test_case("../p03838.py", input_content, expected_output)


def test_problem_p03838_1():
    input_content = "10 20"
    expected_output = "10"
    run_pie_test_case("../p03838.py", input_content, expected_output)


def test_problem_p03838_2():
    input_content = "-10 -20"
    expected_output = "12"
    run_pie_test_case("../p03838.py", input_content, expected_output)


def test_problem_p03838_3():
    input_content = "10 -10"
    expected_output = "1"
    run_pie_test_case("../p03838.py", input_content, expected_output)
