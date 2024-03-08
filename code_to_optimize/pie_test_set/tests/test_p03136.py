from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03136_0():
    input_content = "4\n3 8 5 1"
    expected_output = "Yes"
    run_pie_test_case("../p03136.py", input_content, expected_output)


def test_problem_p03136_1():
    input_content = "4\n3 8 5 1"
    expected_output = "Yes"
    run_pie_test_case("../p03136.py", input_content, expected_output)


def test_problem_p03136_2():
    input_content = "4\n3 8 4 1"
    expected_output = "No"
    run_pie_test_case("../p03136.py", input_content, expected_output)


def test_problem_p03136_3():
    input_content = "10\n1 8 10 5 8 12 34 100 11 3"
    expected_output = "No"
    run_pie_test_case("../p03136.py", input_content, expected_output)
