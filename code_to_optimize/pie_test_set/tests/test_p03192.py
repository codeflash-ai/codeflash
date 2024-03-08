from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03192_0():
    input_content = "1222"
    expected_output = "3"
    run_pie_test_case("../p03192.py", input_content, expected_output)


def test_problem_p03192_1():
    input_content = "9592"
    expected_output = "1"
    run_pie_test_case("../p03192.py", input_content, expected_output)


def test_problem_p03192_2():
    input_content = "3456"
    expected_output = "0"
    run_pie_test_case("../p03192.py", input_content, expected_output)


def test_problem_p03192_3():
    input_content = "1222"
    expected_output = "3"
    run_pie_test_case("../p03192.py", input_content, expected_output)
