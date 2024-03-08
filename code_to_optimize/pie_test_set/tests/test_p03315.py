from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03315_0():
    input_content = "+-++"
    expected_output = "2"
    run_pie_test_case("../p03315.py", input_content, expected_output)


def test_problem_p03315_1():
    input_content = "+-++"
    expected_output = "2"
    run_pie_test_case("../p03315.py", input_content, expected_output)


def test_problem_p03315_2():
    input_content = "-+--"
    expected_output = "-2"
    run_pie_test_case("../p03315.py", input_content, expected_output)


def test_problem_p03315_3():
    input_content = "----"
    expected_output = "-4"
    run_pie_test_case("../p03315.py", input_content, expected_output)
