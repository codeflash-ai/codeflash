from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03675_0():
    input_content = "4\n1 2 3 4"
    expected_output = "4 2 1 3"
    run_pie_test_case("../p03675.py", input_content, expected_output)


def test_problem_p03675_1():
    input_content = "6\n0 6 7 6 7 0"
    expected_output = "0 6 6 0 7 7"
    run_pie_test_case("../p03675.py", input_content, expected_output)


def test_problem_p03675_2():
    input_content = "4\n1 2 3 4"
    expected_output = "4 2 1 3"
    run_pie_test_case("../p03675.py", input_content, expected_output)


def test_problem_p03675_3():
    input_content = "1\n1000000000"
    expected_output = "1000000000"
    run_pie_test_case("../p03675.py", input_content, expected_output)


def test_problem_p03675_4():
    input_content = "3\n1 2 3"
    expected_output = "3 1 2"
    run_pie_test_case("../p03675.py", input_content, expected_output)
