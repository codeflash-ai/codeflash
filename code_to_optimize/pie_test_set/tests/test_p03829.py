from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03829_0():
    input_content = "4 2 5\n1 2 5 7"
    expected_output = "11"
    run_pie_test_case("../p03829.py", input_content, expected_output)


def test_problem_p03829_1():
    input_content = "7 1 100\n40 43 45 105 108 115 124"
    expected_output = "84"
    run_pie_test_case("../p03829.py", input_content, expected_output)


def test_problem_p03829_2():
    input_content = "7 1 2\n24 35 40 68 72 99 103"
    expected_output = "12"
    run_pie_test_case("../p03829.py", input_content, expected_output)


def test_problem_p03829_3():
    input_content = "4 2 5\n1 2 5 7"
    expected_output = "11"
    run_pie_test_case("../p03829.py", input_content, expected_output)
