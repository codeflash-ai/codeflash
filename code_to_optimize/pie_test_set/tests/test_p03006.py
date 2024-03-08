from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03006_0():
    input_content = "2\n1 1\n2 2"
    expected_output = "1"
    run_pie_test_case("../p03006.py", input_content, expected_output)


def test_problem_p03006_1():
    input_content = "3\n1 4\n4 6\n7 8"
    expected_output = "1"
    run_pie_test_case("../p03006.py", input_content, expected_output)


def test_problem_p03006_2():
    input_content = "4\n1 1\n1 2\n2 1\n2 2"
    expected_output = "2"
    run_pie_test_case("../p03006.py", input_content, expected_output)


def test_problem_p03006_3():
    input_content = "2\n1 1\n2 2"
    expected_output = "1"
    run_pie_test_case("../p03006.py", input_content, expected_output)
