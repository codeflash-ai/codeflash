from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02624_0():
    input_content = "4"
    expected_output = "23"
    run_pie_test_case("../p02624.py", input_content, expected_output)


def test_problem_p02624_1():
    input_content = "4"
    expected_output = "23"
    run_pie_test_case("../p02624.py", input_content, expected_output)


def test_problem_p02624_2():
    input_content = "100"
    expected_output = "26879"
    run_pie_test_case("../p02624.py", input_content, expected_output)


def test_problem_p02624_3():
    input_content = "10000000"
    expected_output = "838627288460105"
    run_pie_test_case("../p02624.py", input_content, expected_output)
