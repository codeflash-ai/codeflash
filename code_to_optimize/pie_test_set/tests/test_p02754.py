from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02754_0():
    input_content = "8 3 4"
    expected_output = "4"
    run_pie_test_case("../p02754.py", input_content, expected_output)


def test_problem_p02754_1():
    input_content = "8 0 4"
    expected_output = "0"
    run_pie_test_case("../p02754.py", input_content, expected_output)


def test_problem_p02754_2():
    input_content = "8 3 4"
    expected_output = "4"
    run_pie_test_case("../p02754.py", input_content, expected_output)


def test_problem_p02754_3():
    input_content = "6 2 4"
    expected_output = "2"
    run_pie_test_case("../p02754.py", input_content, expected_output)
