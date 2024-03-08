from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03039_0():
    input_content = "2 2 2"
    expected_output = "8"
    run_pie_test_case("../p03039.py", input_content, expected_output)


def test_problem_p03039_1():
    input_content = "2 2 2"
    expected_output = "8"
    run_pie_test_case("../p03039.py", input_content, expected_output)


def test_problem_p03039_2():
    input_content = "100 100 5000"
    expected_output = "817260251"
    run_pie_test_case("../p03039.py", input_content, expected_output)


def test_problem_p03039_3():
    input_content = "4 5 4"
    expected_output = "87210"
    run_pie_test_case("../p03039.py", input_content, expected_output)
