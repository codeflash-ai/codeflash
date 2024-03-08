from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03126_0():
    input_content = "3 4\n2 1 3\n3 1 2 3\n2 3 2"
    expected_output = "1"
    run_pie_test_case("../p03126.py", input_content, expected_output)


def test_problem_p03126_1():
    input_content = "3 4\n2 1 3\n3 1 2 3\n2 3 2"
    expected_output = "1"
    run_pie_test_case("../p03126.py", input_content, expected_output)


def test_problem_p03126_2():
    input_content = "1 30\n3 5 10 30"
    expected_output = "3"
    run_pie_test_case("../p03126.py", input_content, expected_output)


def test_problem_p03126_3():
    input_content = "5 5\n4 2 3 4 5\n4 1 3 4 5\n4 1 2 4 5\n4 1 2 3 5\n4 1 2 3 4"
    expected_output = "0"
    run_pie_test_case("../p03126.py", input_content, expected_output)
