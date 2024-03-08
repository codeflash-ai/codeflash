from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03773_0():
    input_content = "9 12"
    expected_output = "21"
    run_pie_test_case("../p03773.py", input_content, expected_output)


def test_problem_p03773_1():
    input_content = "9 12"
    expected_output = "21"
    run_pie_test_case("../p03773.py", input_content, expected_output)


def test_problem_p03773_2():
    input_content = "19 0"
    expected_output = "19"
    run_pie_test_case("../p03773.py", input_content, expected_output)


def test_problem_p03773_3():
    input_content = "23 2"
    expected_output = "1"
    run_pie_test_case("../p03773.py", input_content, expected_output)
