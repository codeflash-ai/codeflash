from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03945_0():
    input_content = "BBBWW"
    expected_output = "1"
    run_pie_test_case("../p03945.py", input_content, expected_output)


def test_problem_p03945_1():
    input_content = "WWWWWW"
    expected_output = "0"
    run_pie_test_case("../p03945.py", input_content, expected_output)


def test_problem_p03945_2():
    input_content = "WBWBWBWBWB"
    expected_output = "9"
    run_pie_test_case("../p03945.py", input_content, expected_output)


def test_problem_p03945_3():
    input_content = "BBBWW"
    expected_output = "1"
    run_pie_test_case("../p03945.py", input_content, expected_output)
