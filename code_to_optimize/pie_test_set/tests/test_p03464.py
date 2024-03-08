from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03464_0():
    input_content = "4\n3 4 3 2"
    expected_output = "6 8"
    run_pie_test_case("../p03464.py", input_content, expected_output)


def test_problem_p03464_1():
    input_content = "5\n3 4 100 3 2"
    expected_output = "-1"
    run_pie_test_case("../p03464.py", input_content, expected_output)


def test_problem_p03464_2():
    input_content = "10\n2 2 2 2 2 2 2 2 2 2"
    expected_output = "2 3"
    run_pie_test_case("../p03464.py", input_content, expected_output)


def test_problem_p03464_3():
    input_content = "4\n3 4 3 2"
    expected_output = "6 8"
    run_pie_test_case("../p03464.py", input_content, expected_output)
