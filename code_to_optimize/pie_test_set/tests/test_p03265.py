from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03265_0():
    input_content = "0 0 0 1"
    expected_output = "-1 1 -1 0"
    run_pie_test_case("../p03265.py", input_content, expected_output)


def test_problem_p03265_1():
    input_content = "2 3 6 6"
    expected_output = "3 10 -1 7"
    run_pie_test_case("../p03265.py", input_content, expected_output)


def test_problem_p03265_2():
    input_content = "0 0 0 1"
    expected_output = "-1 1 -1 0"
    run_pie_test_case("../p03265.py", input_content, expected_output)


def test_problem_p03265_3():
    input_content = "31 -41 -59 26"
    expected_output = "-126 -64 -36 -131"
    run_pie_test_case("../p03265.py", input_content, expected_output)
