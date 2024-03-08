from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02791_0():
    input_content = "5\n4 2 5 1 3"
    expected_output = "3"
    run_pie_test_case("../p02791.py", input_content, expected_output)


def test_problem_p02791_1():
    input_content = "8\n5 7 4 2 6 8 1 3"
    expected_output = "4"
    run_pie_test_case("../p02791.py", input_content, expected_output)


def test_problem_p02791_2():
    input_content = "4\n4 3 2 1"
    expected_output = "4"
    run_pie_test_case("../p02791.py", input_content, expected_output)


def test_problem_p02791_3():
    input_content = "6\n1 2 3 4 5 6"
    expected_output = "1"
    run_pie_test_case("../p02791.py", input_content, expected_output)


def test_problem_p02791_4():
    input_content = "5\n4 2 5 1 3"
    expected_output = "3"
    run_pie_test_case("../p02791.py", input_content, expected_output)


def test_problem_p02791_5():
    input_content = "1\n1"
    expected_output = "1"
    run_pie_test_case("../p02791.py", input_content, expected_output)
