from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02576_0():
    input_content = "20 12 6"
    expected_output = "12"
    run_pie_test_case("../p02576.py", input_content, expected_output)


def test_problem_p02576_1():
    input_content = "1000 1 1000"
    expected_output = "1000000"
    run_pie_test_case("../p02576.py", input_content, expected_output)


def test_problem_p02576_2():
    input_content = "20 12 6"
    expected_output = "12"
    run_pie_test_case("../p02576.py", input_content, expected_output)
