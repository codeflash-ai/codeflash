from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03474_0():
    input_content = "3 4\n269-6650"
    expected_output = "Yes"
    run_pie_test_case("../p03474.py", input_content, expected_output)


def test_problem_p03474_1():
    input_content = "1 2\n7444"
    expected_output = "No"
    run_pie_test_case("../p03474.py", input_content, expected_output)


def test_problem_p03474_2():
    input_content = "3 4\n269-6650"
    expected_output = "Yes"
    run_pie_test_case("../p03474.py", input_content, expected_output)


def test_problem_p03474_3():
    input_content = "1 1\n---"
    expected_output = "No"
    run_pie_test_case("../p03474.py", input_content, expected_output)
