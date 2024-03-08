from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03394_0():
    input_content = "3"
    expected_output = "2 5 63"
    run_pie_test_case("../p03394.py", input_content, expected_output)


def test_problem_p03394_1():
    input_content = "3"
    expected_output = "2 5 63"
    run_pie_test_case("../p03394.py", input_content, expected_output)


def test_problem_p03394_2():
    input_content = "4"
    expected_output = "2 5 20 63"
    run_pie_test_case("../p03394.py", input_content, expected_output)
