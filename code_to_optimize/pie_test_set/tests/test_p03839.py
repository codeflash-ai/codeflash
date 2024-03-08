from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03839_0():
    input_content = "5 3\n-10 10 -10 10 -10"
    expected_output = "10"
    run_pie_test_case("../p03839.py", input_content, expected_output)


def test_problem_p03839_1():
    input_content = "10 5\n5 -4 -5 -8 -4 7 2 -4 0 7"
    expected_output = "17"
    run_pie_test_case("../p03839.py", input_content, expected_output)


def test_problem_p03839_2():
    input_content = "1 1\n-10"
    expected_output = "0"
    run_pie_test_case("../p03839.py", input_content, expected_output)


def test_problem_p03839_3():
    input_content = "5 3\n-10 10 -10 10 -10"
    expected_output = "10"
    run_pie_test_case("../p03839.py", input_content, expected_output)


def test_problem_p03839_4():
    input_content = "4 2\n10 -10 -10 10"
    expected_output = "20"
    run_pie_test_case("../p03839.py", input_content, expected_output)
