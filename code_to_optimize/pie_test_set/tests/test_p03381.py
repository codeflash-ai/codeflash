from code_to_optimize.pie_test_set.scripts.run_pie_test_case import run_pie_test_case


def test_problem_p03381_0():
    input_content = "4\n2 4 4 3"
    expected_output = "4\n3\n3\n4"
    run_pie_test_case("../p03381.py", input_content, expected_output)


def test_problem_p03381_1():
    input_content = "6\n5 5 4 4 3 3"
    expected_output = "4\n4\n4\n4\n4\n4"
    run_pie_test_case("../p03381.py", input_content, expected_output)


def test_problem_p03381_2():
    input_content = "4\n2 4 4 3"
    expected_output = "4\n3\n3\n4"
    run_pie_test_case("../p03381.py", input_content, expected_output)


def test_problem_p03381_3():
    input_content = "2\n1 2"
    expected_output = "2\n1"
    run_pie_test_case("../p03381.py", input_content, expected_output)
