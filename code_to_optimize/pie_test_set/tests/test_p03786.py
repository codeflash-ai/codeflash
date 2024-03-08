from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03786_0():
    input_content = "3\n3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03786.py", input_content, expected_output)


def test_problem_p03786_1():
    input_content = "5\n1 1 1 1 1"
    expected_output = "5"
    run_pie_test_case("../p03786.py", input_content, expected_output)


def test_problem_p03786_2():
    input_content = "3\n3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03786.py", input_content, expected_output)


def test_problem_p03786_3():
    input_content = "6\n40 1 30 2 7 20"
    expected_output = "4"
    run_pie_test_case("../p03786.py", input_content, expected_output)
