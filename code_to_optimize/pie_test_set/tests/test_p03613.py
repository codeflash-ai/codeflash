from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03613_0():
    input_content = "7\n3 1 4 1 5 9 2"
    expected_output = "4"
    run_pie_test_case("../p03613.py", input_content, expected_output)


def test_problem_p03613_1():
    input_content = "7\n3 1 4 1 5 9 2"
    expected_output = "4"
    run_pie_test_case("../p03613.py", input_content, expected_output)


def test_problem_p03613_2():
    input_content = "10\n0 1 2 3 4 5 6 7 8 9"
    expected_output = "3"
    run_pie_test_case("../p03613.py", input_content, expected_output)


def test_problem_p03613_3():
    input_content = "1\n99999"
    expected_output = "1"
    run_pie_test_case("../p03613.py", input_content, expected_output)
