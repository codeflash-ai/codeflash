from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03556_0():
    input_content = "10"
    expected_output = "9"
    run_pie_test_case("../p03556.py", input_content, expected_output)


def test_problem_p03556_1():
    input_content = "81"
    expected_output = "81"
    run_pie_test_case("../p03556.py", input_content, expected_output)


def test_problem_p03556_2():
    input_content = "10"
    expected_output = "9"
    run_pie_test_case("../p03556.py", input_content, expected_output)


def test_problem_p03556_3():
    input_content = "271828182"
    expected_output = "271821169"
    run_pie_test_case("../p03556.py", input_content, expected_output)
