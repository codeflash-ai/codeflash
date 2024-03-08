from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03012_0():
    input_content = "3\n1 2 3"
    expected_output = "0"
    run_pie_test_case("../p03012.py", input_content, expected_output)


def test_problem_p03012_1():
    input_content = "3\n1 2 3"
    expected_output = "0"
    run_pie_test_case("../p03012.py", input_content, expected_output)


def test_problem_p03012_2():
    input_content = "4\n1 3 1 1"
    expected_output = "2"
    run_pie_test_case("../p03012.py", input_content, expected_output)


def test_problem_p03012_3():
    input_content = "8\n27 23 76 2 3 5 62 52"
    expected_output = "2"
    run_pie_test_case("../p03012.py", input_content, expected_output)
