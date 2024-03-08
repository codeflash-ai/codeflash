from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p02317_0():
    input_content = "5\n5\n1\n3\n2\n4"
    expected_output = "3"
    run_pie_test_case("../p02317.py", input_content, expected_output)


def test_problem_p02317_1():
    input_content = "5\n5\n1\n3\n2\n4"
    expected_output = "3"
    run_pie_test_case("../p02317.py", input_content, expected_output)


def test_problem_p02317_2():
    input_content = "3\n1\n1\n1"
    expected_output = "1"
    run_pie_test_case("../p02317.py", input_content, expected_output)
