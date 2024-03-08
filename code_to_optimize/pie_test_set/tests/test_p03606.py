from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03606_0():
    input_content = "1\n24 30"
    expected_output = "7"
    run_pie_test_case("../p03606.py", input_content, expected_output)


def test_problem_p03606_1():
    input_content = "1\n24 30"
    expected_output = "7"
    run_pie_test_case("../p03606.py", input_content, expected_output)


def test_problem_p03606_2():
    input_content = "2\n6 8\n3 3"
    expected_output = "4"
    run_pie_test_case("../p03606.py", input_content, expected_output)
