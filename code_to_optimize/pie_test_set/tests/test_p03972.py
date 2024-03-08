from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03972_0():
    input_content = "2 2\n3\n5\n2\n7"
    expected_output = "29"
    run_pie_test_case("../p03972.py", input_content, expected_output)


def test_problem_p03972_1():
    input_content = "4 3\n2\n4\n8\n1\n2\n9\n3"
    expected_output = "60"
    run_pie_test_case("../p03972.py", input_content, expected_output)


def test_problem_p03972_2():
    input_content = "2 2\n3\n5\n2\n7"
    expected_output = "29"
    run_pie_test_case("../p03972.py", input_content, expected_output)
