from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02554_0():
    input_content = "2"
    expected_output = "2"
    run_pie_test_case("../p02554.py", input_content, expected_output)


def test_problem_p02554_1():
    input_content = "869121"
    expected_output = "2511445"
    run_pie_test_case("../p02554.py", input_content, expected_output)


def test_problem_p02554_2():
    input_content = "1"
    expected_output = "0"
    run_pie_test_case("../p02554.py", input_content, expected_output)


def test_problem_p02554_3():
    input_content = "2"
    expected_output = "2"
    run_pie_test_case("../p02554.py", input_content, expected_output)
