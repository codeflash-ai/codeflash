from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03062_0():
    input_content = "3\n-10 5 -4"
    expected_output = "19"
    run_pie_test_case("../p03062.py", input_content, expected_output)


def test_problem_p03062_1():
    input_content = "11\n-1000000000 1000000000 -1000000000 1000000000 -1000000000 0 1000000000 -1000000000 1000000000 -1000000000 1000000000"
    expected_output = "10000000000"
    run_pie_test_case("../p03062.py", input_content, expected_output)


def test_problem_p03062_2():
    input_content = "5\n10 -4 -8 -11 3"
    expected_output = "30"
    run_pie_test_case("../p03062.py", input_content, expected_output)


def test_problem_p03062_3():
    input_content = "3\n-10 5 -4"
    expected_output = "19"
    run_pie_test_case("../p03062.py", input_content, expected_output)
