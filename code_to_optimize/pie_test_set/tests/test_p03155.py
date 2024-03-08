from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03155_0():
    input_content = "3\n2\n3"
    expected_output = "2"
    run_pie_test_case("../p03155.py", input_content, expected_output)


def test_problem_p03155_1():
    input_content = "5\n4\n2"
    expected_output = "8"
    run_pie_test_case("../p03155.py", input_content, expected_output)


def test_problem_p03155_2():
    input_content = "3\n2\n3"
    expected_output = "2"
    run_pie_test_case("../p03155.py", input_content, expected_output)


def test_problem_p03155_3():
    input_content = "100\n1\n1"
    expected_output = "10000"
    run_pie_test_case("../p03155.py", input_content, expected_output)
