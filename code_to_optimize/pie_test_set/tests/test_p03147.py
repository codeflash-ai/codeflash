from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03147_0():
    input_content = "4\n1 2 2 1"
    expected_output = "2"
    run_pie_test_case("../p03147.py", input_content, expected_output)


def test_problem_p03147_1():
    input_content = "4\n1 2 2 1"
    expected_output = "2"
    run_pie_test_case("../p03147.py", input_content, expected_output)


def test_problem_p03147_2():
    input_content = "8\n4 23 75 0 23 96 50 100"
    expected_output = "221"
    run_pie_test_case("../p03147.py", input_content, expected_output)


def test_problem_p03147_3():
    input_content = "5\n3 1 2 3 1"
    expected_output = "5"
    run_pie_test_case("../p03147.py", input_content, expected_output)
