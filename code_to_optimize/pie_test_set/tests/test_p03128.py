from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03128_0():
    input_content = "20 4\n3 7 8 4"
    expected_output = "777773"
    run_pie_test_case("../p03128.py", input_content, expected_output)


def test_problem_p03128_1():
    input_content = "101 9\n9 8 7 6 5 4 3 2 1"
    expected_output = "71111111111111111111111111111111111111111111111111"
    run_pie_test_case("../p03128.py", input_content, expected_output)


def test_problem_p03128_2():
    input_content = "20 4\n3 7 8 4"
    expected_output = "777773"
    run_pie_test_case("../p03128.py", input_content, expected_output)


def test_problem_p03128_3():
    input_content = "15 3\n5 4 6"
    expected_output = "654"
    run_pie_test_case("../p03128.py", input_content, expected_output)
