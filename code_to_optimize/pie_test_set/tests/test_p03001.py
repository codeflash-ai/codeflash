from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03001_0():
    input_content = "2 3 1 2"
    expected_output = "3.000000 0"
    run_pie_test_case("../p03001.py", input_content, expected_output)


def test_problem_p03001_1():
    input_content = "2 3 1 2"
    expected_output = "3.000000 0"
    run_pie_test_case("../p03001.py", input_content, expected_output)


def test_problem_p03001_2():
    input_content = "2 2 1 1"
    expected_output = "2.000000 1"
    run_pie_test_case("../p03001.py", input_content, expected_output)
