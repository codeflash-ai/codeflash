from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03434_0():
    input_content = "2\n3 1"
    expected_output = "2"
    run_pie_test_case("../p03434.py", input_content, expected_output)


def test_problem_p03434_1():
    input_content = "4\n20 18 2 18"
    expected_output = "18"
    run_pie_test_case("../p03434.py", input_content, expected_output)


def test_problem_p03434_2():
    input_content = "2\n3 1"
    expected_output = "2"
    run_pie_test_case("../p03434.py", input_content, expected_output)


def test_problem_p03434_3():
    input_content = "3\n2 7 4"
    expected_output = "5"
    run_pie_test_case("../p03434.py", input_content, expected_output)
