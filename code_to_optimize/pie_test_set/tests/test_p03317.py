from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03317_0():
    input_content = "4 3\n2 3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03317.py", input_content, expected_output)


def test_problem_p03317_1():
    input_content = "8 3\n7 3 1 8 4 6 2 5"
    expected_output = "4"
    run_pie_test_case("../p03317.py", input_content, expected_output)


def test_problem_p03317_2():
    input_content = "4 3\n2 3 1 4"
    expected_output = "2"
    run_pie_test_case("../p03317.py", input_content, expected_output)


def test_problem_p03317_3():
    input_content = "3 3\n1 2 3"
    expected_output = "1"
    run_pie_test_case("../p03317.py", input_content, expected_output)
