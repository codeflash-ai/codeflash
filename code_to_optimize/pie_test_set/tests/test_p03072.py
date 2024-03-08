from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03072_0():
    input_content = "4\n6 5 6 8"
    expected_output = "3"
    run_pie_test_case("../p03072.py", input_content, expected_output)


def test_problem_p03072_1():
    input_content = "4\n6 5 6 8"
    expected_output = "3"
    run_pie_test_case("../p03072.py", input_content, expected_output)


def test_problem_p03072_2():
    input_content = "5\n9 5 6 8 4"
    expected_output = "1"
    run_pie_test_case("../p03072.py", input_content, expected_output)


def test_problem_p03072_3():
    input_content = "5\n4 5 3 5 4"
    expected_output = "3"
    run_pie_test_case("../p03072.py", input_content, expected_output)
