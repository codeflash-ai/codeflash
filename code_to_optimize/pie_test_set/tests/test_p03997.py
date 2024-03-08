from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03997_0():
    input_content = "3\n4\n2"
    expected_output = "7"
    run_pie_test_case("../p03997.py", input_content, expected_output)


def test_problem_p03997_1():
    input_content = "4\n4\n4"
    expected_output = "16"
    run_pie_test_case("../p03997.py", input_content, expected_output)


def test_problem_p03997_2():
    input_content = "3\n4\n2"
    expected_output = "7"
    run_pie_test_case("../p03997.py", input_content, expected_output)
