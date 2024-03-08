from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03894_0():
    input_content = "10 3\n1 3\n2 4\n4 5"
    expected_output = "4"
    run_pie_test_case("../p03894.py", input_content, expected_output)


def test_problem_p03894_1():
    input_content = "10 3\n1 3\n2 4\n4 5"
    expected_output = "4"
    run_pie_test_case("../p03894.py", input_content, expected_output)


def test_problem_p03894_2():
    input_content = "20 3\n1 7\n8 20\n1 19"
    expected_output = "5"
    run_pie_test_case("../p03894.py", input_content, expected_output)
