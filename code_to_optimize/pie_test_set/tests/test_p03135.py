from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03135_0():
    input_content = "8 3"
    expected_output = "2.6666666667"
    run_pie_test_case("../p03135.py", input_content, expected_output)


def test_problem_p03135_1():
    input_content = "1 100"
    expected_output = "0.0100000000"
    run_pie_test_case("../p03135.py", input_content, expected_output)


def test_problem_p03135_2():
    input_content = "99 1"
    expected_output = "99.0000000000"
    run_pie_test_case("../p03135.py", input_content, expected_output)


def test_problem_p03135_3():
    input_content = "8 3"
    expected_output = "2.6666666667"
    run_pie_test_case("../p03135.py", input_content, expected_output)
