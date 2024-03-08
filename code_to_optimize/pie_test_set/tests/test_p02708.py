from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02708_0():
    input_content = "3 2"
    expected_output = "10"
    run_pie_test_case("../p02708.py", input_content, expected_output)


def test_problem_p02708_1():
    input_content = "141421 35623"
    expected_output = "220280457"
    run_pie_test_case("../p02708.py", input_content, expected_output)


def test_problem_p02708_2():
    input_content = "200000 200001"
    expected_output = "1"
    run_pie_test_case("../p02708.py", input_content, expected_output)


def test_problem_p02708_3():
    input_content = "3 2"
    expected_output = "10"
    run_pie_test_case("../p02708.py", input_content, expected_output)
