from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03238_0():
    input_content = "1"
    expected_output = "Hello World"
    run_pie_test_case("../p03238.py", input_content, expected_output)


def test_problem_p03238_1():
    input_content = "1"
    expected_output = "Hello World"
    run_pie_test_case("../p03238.py", input_content, expected_output)


def test_problem_p03238_2():
    input_content = "2\n3\n5"
    expected_output = "8"
    run_pie_test_case("../p03238.py", input_content, expected_output)
