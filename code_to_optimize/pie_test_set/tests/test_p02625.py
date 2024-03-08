from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02625_0():
    input_content = "2 2"
    expected_output = "2"
    run_pie_test_case("../p02625.py", input_content, expected_output)


def test_problem_p02625_1():
    input_content = "141421 356237"
    expected_output = "881613484"
    run_pie_test_case("../p02625.py", input_content, expected_output)


def test_problem_p02625_2():
    input_content = "2 2"
    expected_output = "2"
    run_pie_test_case("../p02625.py", input_content, expected_output)


def test_problem_p02625_3():
    input_content = "2 3"
    expected_output = "18"
    run_pie_test_case("../p02625.py", input_content, expected_output)
