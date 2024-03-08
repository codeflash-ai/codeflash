from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02971_0():
    input_content = "3\n1\n4\n3"
    expected_output = "4\n3\n4"
    run_pie_test_case("../p02971.py", input_content, expected_output)


def test_problem_p02971_1():
    input_content = "2\n5\n5"
    expected_output = "5\n5"
    run_pie_test_case("../p02971.py", input_content, expected_output)


def test_problem_p02971_2():
    input_content = "3\n1\n4\n3"
    expected_output = "4\n3\n4"
    run_pie_test_case("../p02971.py", input_content, expected_output)
