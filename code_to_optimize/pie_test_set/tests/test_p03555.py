from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03555_0():
    input_content = "pot\ntop"
    expected_output = "YES"
    run_pie_test_case("../p03555.py", input_content, expected_output)


def test_problem_p03555_1():
    input_content = "tab\nbet"
    expected_output = "NO"
    run_pie_test_case("../p03555.py", input_content, expected_output)


def test_problem_p03555_2():
    input_content = "eye\neel"
    expected_output = "NO"
    run_pie_test_case("../p03555.py", input_content, expected_output)


def test_problem_p03555_3():
    input_content = "pot\ntop"
    expected_output = "YES"
    run_pie_test_case("../p03555.py", input_content, expected_output)
