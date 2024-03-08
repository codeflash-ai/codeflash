from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02741_0():
    input_content = "6"
    expected_output = "2"
    run_pie_test_case("../p02741.py", input_content, expected_output)


def test_problem_p02741_1():
    input_content = "27"
    expected_output = "5"
    run_pie_test_case("../p02741.py", input_content, expected_output)


def test_problem_p02741_2():
    input_content = "6"
    expected_output = "2"
    run_pie_test_case("../p02741.py", input_content, expected_output)
