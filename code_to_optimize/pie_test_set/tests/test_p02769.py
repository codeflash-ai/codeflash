from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02769_0():
    input_content = "3 2"
    expected_output = "10"
    run_pie_test_case("../p02769.py", input_content, expected_output)


def test_problem_p02769_1():
    input_content = "3 2"
    expected_output = "10"
    run_pie_test_case("../p02769.py", input_content, expected_output)


def test_problem_p02769_2():
    input_content = "200000 1000000000"
    expected_output = "607923868"
    run_pie_test_case("../p02769.py", input_content, expected_output)


def test_problem_p02769_3():
    input_content = "15 6"
    expected_output = "22583772"
    run_pie_test_case("../p02769.py", input_content, expected_output)
