from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03745_0():
    input_content = "6\n1 2 3 2 2 1"
    expected_output = "2"
    run_pie_test_case("../p03745.py", input_content, expected_output)


def test_problem_p03745_1():
    input_content = "6\n1 2 3 2 2 1"
    expected_output = "2"
    run_pie_test_case("../p03745.py", input_content, expected_output)


def test_problem_p03745_2():
    input_content = "9\n1 2 1 2 1 2 1 2 1"
    expected_output = "5"
    run_pie_test_case("../p03745.py", input_content, expected_output)


def test_problem_p03745_3():
    input_content = "7\n1 2 3 2 1 999999999 1000000000"
    expected_output = "3"
    run_pie_test_case("../p03745.py", input_content, expected_output)
