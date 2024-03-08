from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02694_0():
    input_content = "103"
    expected_output = "3"
    run_pie_test_case("../p02694.py", input_content, expected_output)


def test_problem_p02694_1():
    input_content = "1333333333"
    expected_output = "1706"
    run_pie_test_case("../p02694.py", input_content, expected_output)


def test_problem_p02694_2():
    input_content = "103"
    expected_output = "3"
    run_pie_test_case("../p02694.py", input_content, expected_output)


def test_problem_p02694_3():
    input_content = "1000000000000000000"
    expected_output = "3760"
    run_pie_test_case("../p02694.py", input_content, expected_output)
