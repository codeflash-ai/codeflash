from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03391_0():
    input_content = "2\n1 2\n3 2"
    expected_output = "2"
    run_pie_test_case("../p03391.py", input_content, expected_output)


def test_problem_p03391_1():
    input_content = "3\n8 3\n0 1\n4 8"
    expected_output = "9"
    run_pie_test_case("../p03391.py", input_content, expected_output)


def test_problem_p03391_2():
    input_content = "1\n1 1"
    expected_output = "0"
    run_pie_test_case("../p03391.py", input_content, expected_output)


def test_problem_p03391_3():
    input_content = "2\n1 2\n3 2"
    expected_output = "2"
    run_pie_test_case("../p03391.py", input_content, expected_output)
